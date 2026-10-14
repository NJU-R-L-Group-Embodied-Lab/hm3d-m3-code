from openai import OpenAI
from PIL import Image
from habitat import logger
import io
import base64
import os
import json
import numpy as np

DEFAULT_LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_LLM_MODEL = "qwen3.5-27b"


def _first_env(*names):
    """Return the first non-empty environment variable among ``names`` (or '')."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return ""


def env_llm_api_key(*extra_fallbacks):
    """API key from LLM_API_KEY, falling back to DASHSCOPE_API_KEY, the given
    extra names, and LAVIRA_API_KEY. Keys are only ever read from the environment."""
    return _first_env("LLM_API_KEY", "DASHSCOPE_API_KEY", *extra_fallbacks, "LAVIRA_API_KEY")


def env_llm_base_url():
    """OpenAI-compatible endpoint from LLM_BASE_URL / DASHSCOPE_BASE_URL."""
    return _first_env("LLM_BASE_URL", "DASHSCOPE_BASE_URL") or DEFAULT_LLM_BASE_URL


def env_llm_model():
    """Model name from LLM_MODEL (default qwen3.5-27b)."""
    return _first_env("LLM_MODEL") or DEFAULT_LLM_MODEL


class LaViRA_API:

    def __init__(self, la_api_key=None, la_base_url=None, la_model_name="qwen3.5-27b",
                 va_model_name=None, va_api_key=None, va_base_url=None,
                 la_enable_thinking=False, la_thinking_budget=0):
        # Default endpoint: Aliyun Bailian / DashScope OpenAI-compatible mode, qwen3.5-27b.
        self.la_enable_thinking = la_enable_thinking
        self.la_thinking_budget = la_thinking_budget
        self.la_client = OpenAI(
            api_key=la_api_key,
            base_url=la_base_url,
            timeout=2000
        )
        self.la_model_name = la_model_name

        if va_model_name:
            self.va_client = OpenAI(
                api_key=va_api_key,
                base_url=va_base_url,
                timeout=2000

            )
            self.va_model_name = va_model_name
        else:
            self.va_client = None
            self.va_model_name = None

        # Usage statistics
        self.reset_stats()

    def image_to_base64(self, image):
        """Encode a PIL Image or numpy array as base64."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _save_debug_info(self, log_path, messages, response_text):
        if not log_path:
            return
        
        try:
            os.makedirs(log_path, exist_ok=True)
            
            # Save images and create a clean message list for saving
            saved_messages = []
            img_count = 0
            
            for msg in messages:
                new_msg = {'role': msg['role'], 'content': []}
                if isinstance(msg['content'], list):
                    for item in msg['content']:
                        if isinstance(item, dict) and item.get('type') == 'image_url':
                            url = item['image_url']['url']
                            if url.startswith('data:image/'):
                                try:
                                    # Extract base64
                                    header, encoded = url.split(',', 1)
                                    data = base64.b64decode(encoded)
                                    img_filename = f"image_{img_count}.png"
                                    img_path = os.path.join(log_path, img_filename)
                                    with open(img_path, 'wb') as f:
                                        f.write(data)
                                    
                                    new_msg['content'].append({
                                        'type': 'image_url',
                                        'image_url': {'url': img_filename}
                                    })
                                    img_count += 1
                                except Exception as e:
                                    logger.error(f"Failed to save image: {e}")
                                    new_msg['content'].append({'type': 'image_url', 'image_url': {'url': 'FAILED_TO_SAVE'}})
                            else:
                                new_msg['content'].append(item)
                        else:
                            new_msg['content'].append(item)
                else:
                     new_msg['content'] = msg['content']
                saved_messages.append(new_msg)
                
            # Save prompt text
            with open(os.path.join(log_path, 'prompt.json'), 'w') as f:
                json.dump(saved_messages, f, indent=2)
                
            # Save response
            with open(os.path.join(log_path, 'response.txt'), 'w') as f:
                f.write(str(response_text))
        except Exception as e:
            logger.error(f"Failed to save debug info: {e}")

    def generate(self, messages, images=None, max_new_tokens=1024, temperature=0.7, use_la=False, log_path=None, retries=0, max_retries=5, **kwargs):
        """
        Mimics the original model.generate interface.
        Args:
            messages: list of text messages
            images: list of images
            max_new_tokens: max tokens to generate
            temperature: sampling temperature
            use_la: use the second (LA) model
            log_path: Path to save debug logs (images and prompt)
            retries: Current retry count
            max_retries: Maximum number of retries
        """
        # use_la = False
        import time
        t = time.time()
        # Pick client and model
        if use_la and self.la_client:
            client = self.la_client
            model_name = self.la_model_name
            stats_key = 'Language Action Model'
        else:
            client = self.va_client
            model_name = self.va_model_name
            stats_key = 'Vision Action Model'

        # Default: disable thinking on the Qwen3 family for both LA and VA paths.
        # When la_enable_thinking is set on the API client, override for LA path
        # only and add thinking_budget. VA always stays thinking-off.
        extra_body = kwargs.pop('extra_body', None) or {}
        if stats_key == 'Language Action Model' and self.la_enable_thinking:
            extra_body.setdefault('enable_thinking', True)
            if self.la_thinking_budget and self.la_thinking_budget > 0:
                extra_body.setdefault('thinking_budget', int(self.la_thinking_budget))
        else:
            extra_body.setdefault('enable_thinking', False)

        try:
            if stats_key == 'Language Action Model':
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_completion_tokens=max_new_tokens,
                    timeout=120,
                    extra_body=extra_body,
                    **kwargs
                )
            else:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    timeout=120,
                    reasoning_effort='low',
                    extra_body=extra_body,
                    **kwargs
                )
            logger.info(response)
            # Handle case where response is a string (e.g. from some proxies or raw returns)
            if isinstance(response, str):
                logger.info(f"API returned string response for {model_name}")
                self.stats[stats_key]['calls'] += 1
                self._save_debug_info(log_path, messages, response)
                return response

            # Update statistics
            self.stats[stats_key]['calls'] += 1
            if hasattr(response, 'usage') and response.usage:
                logger.info(f"API Call usage - {response.usage}")
                self.stats[stats_key]['input_tokens'] += response.usage.prompt_tokens or 0
                self.stats[stats_key]['output_tokens'] += response.usage.completion_tokens or 0
                self.stats[stats_key]['total_tokens'] += response.usage.total_tokens or 0

                logger.info(f"{stats_key.upper()} model usage - Input: {response.usage.prompt_tokens}, "
                            f"Output: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")

            logger.info(f'Generating uses {time.time() - t} seconds.')
            content = response.choices[0].message.content
            self._save_debug_info(log_path, messages, content)
            return content

        except Exception as e:
            err_str = str(e)
            logger.error(f"API Call error with {'secondary' if use_la else 'primary'} model ({model_name}): {e}")
            # Non-recoverable errors — retrying won't help; short-circuit to keep run time bounded.
            non_recoverable = (
                'data_inspection_failed' in err_str or
                'DataInspectionFailed' in err_str or
                'inappropriate content' in err_str or
                'invalid_request_error' in err_str
            )
            if non_recoverable:
                logger.error(f"Non-recoverable error detected; skipping retries.")
                return "Error: API rejected request (non-recoverable)"
            if retries >= max_retries:
                logger.error(f"Max retries ({max_retries}) reached. Giving up.")
                return "Error: Failed to get response from API after max retries"

            logger.info(f'Forcing retry ({retries + 1}/{max_retries})..')
            import time
            time.sleep(30)
            return self.generate(messages, images, max_new_tokens, temperature, use_la, log_path=log_path, retries=retries + 1, max_retries=max_retries, **kwargs)

    def get_model_info(self):
        """Return model info."""
        info = {
            "primary_model": self.model_name,
            "secondary_model": self.la_model_name if self.secondary_client else None,
            "has_secondary": self.secondary_client is not None
        }
        return info

    def get_usage_stats(self):
        """Return usage statistics."""
        return self.stats.copy()

    def print_usage_stats(self):
        """Print detailed usage statistics."""
        total_calls = self.stats['Language Action Model']['calls'] + self.stats['Vision Action Model']['calls']
        total_tokens = self.stats['Language Action Model']['total_tokens'] + self.stats['Vision Action Model']['total_tokens']
        if self.la_client:
            logger.info("=== MODEL USAGE STATISTICS ===")
            logger.info(f"Language Action Model ({self.la_model_name}):")
            logger.info(f"  - Calls: {self.stats['Language Action Model']['calls']}")
            logger.info(f"  - Input tokens: {self.stats['Language Action Model']['input_tokens']:,}")
            logger.info(f"  - Output tokens: {self.stats['Language Action Model']['output_tokens']:,}")
            logger.info(f"  - Total tokens: {self.stats['Language Action Model']['total_tokens']:,}")

        if self.va_client:
            logger.info(f"Vision Action Model ({self.va_model_name}):")
            logger.info(f"  - Calls: {self.stats['Vision Action Model']['calls']}")
            logger.info(f"  - Input tokens: {self.stats['Vision Action Model']['input_tokens']:,}")
            logger.info(f"  - Output tokens: {self.stats['Vision Action Model']['output_tokens']:,}")
            logger.info(f"  - Total tokens: {self.stats['Vision Action Model']['total_tokens']:,}")

        logger.info(f"TOTAL:")
        logger.info(f"  - Total calls: {total_calls}")
        logger.info(f"  - Total tokens: {total_tokens:,}")
        logger.info("===============================")

        return {
            'la': self.stats['Language Action Model'].copy(),
            'va': self.stats['Vision Action Model'].copy(),
            'total_calls': total_calls,
            'total_tokens': total_tokens
        }

    def reset_stats(self):
        """Reset statistics."""
        self.stats = {
            'Language Action Model': {
                'calls': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            },
            'Vision Action Model': {
                'calls': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }
        }

    def eval(self):
        """Compatibility method."""
        pass
