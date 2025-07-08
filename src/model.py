"""
Model classes for different LLM providers and utilities.
"""
import os
import logging
import torch
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def get_response(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Get response from the model."""
        pass


class OpenSourceModel(BaseModel):
    """Class for open-source models like Qwen and Mistral."""
    
    def __init__(self, model_name: str):
        """Initialize the open-source model."""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. This script requires a GPU.")

        # Configure quantization for memory efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False, #debugging
            bnb_4bit_quant_type="nf4"
        )
        
        # Load tokenizer and model with safety checks
        logger.info(f"Loading model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map={"": 0},  # setting to "auto"/"balanced" causes the script to hang
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                offload_folder="offload_folder"
            )
            
            self.model.eval()
            
            # Get model's maximum context length
            if hasattr(self.model.config, 'max_sequence_length'):
                self.max_model_length = self.model.config.max_sequence_length
            elif hasattr(self.model.config, 'max_position_embeddings'):
                self.max_model_length = self.model.config.max_position_embeddings
            else:
                self.max_model_length = 2048  # Conservative default
            
            self.max_model_length = min(self.max_model_length, 2048)
            logger.info(f"Model maximum sequence length set to: {self.max_model_length}")
            
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def get_response(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Get response from the open-source model."""
        try:
            torch.cuda.empty_cache()
            
            max_input_length = self.max_model_length - max_new_tokens - 100

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
                add_special_tokens=True
            )

            input_device = next(self.model.parameters()).device
            inputs = {k: v.to(input_device) for k, v in inputs.items()}
            
            try:
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("CUDA out of memory. Retrying with a smaller input sequence.")
                    torch.cuda.empty_cache()
                    
                    new_max_length = inputs['input_ids'].shape[1] // 2
                    inputs['input_ids'] = inputs['input_ids'][:, -new_max_length:]
                    inputs['attention_mask'] = inputs['attention_mask'][:, -new_max_length:]
                    
                    with torch.inference_mode():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            num_beams=1,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            use_cache=True
                        )
                else:
                    raise

            input_token_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_token_len:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            torch.cuda.empty_cache()
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error in model query: {str(e)}")
            if "CUDA" in str(e):
                logger.error("A CUDA error occurred. Clearing cache.")
                torch.cuda.empty_cache()
            return ""


class LangChainModel(BaseModel):
    """Class for LangChain-based models (GPT-4, Gemini, Claude)."""
    
    def __init__(self, model_name: str):
        """Initialize the LangChain model."""
        self.model_name = model_name
        
        try:
            from langchain_openai import ChatOpenAI
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            logger.error(f"LangChain import error: {e}")
            raise ImportError(
                "LangChain packages are required for API-based models. Install with: "
                "pip install langchain-openai langchain-google-genai langchain-anthropic"
            )
        
        try:
            if model_name == "gpt-4":
                api_key = os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in environment variables")
                logger.info(f"Initializing OpenAI GPT-4 model...")
                self.model = ChatOpenAI(
                    model="gpt-4-turbo-preview",
                    temperature=0,
                    max_tokens=150,
                    openai_api_key=api_key
                )
                logger.info(f"GPT-4 model initialized successfully")
            elif model_name == "gemini":
                api_key = os.getenv('GOOGLE_API_KEY')
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY not found in environment variables")
                logger.info(f"Initializing Google Gemini model...")
                self.model = ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    temperature=0,
                    max_output_tokens=150, 
                    google_api_key=api_key
                )
                logger.info(f"Gemini model initialized successfully")
            elif model_name == "claude":
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
                logger.info(f"Initializing Anthropic Claude model...")
                self.model = ChatAnthropic(
                    model="claude-3-opus-20240229",
                    temperature=0,
                    max_tokens=150,
                    anthropic_api_key=api_key
                )
                logger.info(f"Claude model initialized successfully")
            else:
                raise ValueError(f"Unsupported LangChain model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing {model_name} model: {e}")
            raise

    def get_response(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Get response from the LangChain model."""
        try:
            # Simple invoke - max_tokens is already set in constructor
            response = self.model.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error in {self.model_name} query: {str(e)}")
            logger.error(f"Full error details: {e}")
            return ""


def get_model(model_name: str) -> BaseModel:
    """Factory function to get the appropriate model instance.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        An instance of the appropriate model class
        
    Raises:
        ValueError: If the model name is not supported
        ImportError: If the required package for the selected model is not installed
    """
    if model_name in ["Qwen/Qwen3-8B", "mistralai/Mistral-7B-Instruct-v0.2"]:
        return OpenSourceModel(model_name)
    elif model_name in ["gpt-4", "gemini", "claude"]:
        return LangChainModel(model_name)
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            "Choose from: Qwen/Qwen3-8B, mistralai/Mistral-7B-Instruct-v0.2, "
            "gpt-4, gemini, claude"
        )
