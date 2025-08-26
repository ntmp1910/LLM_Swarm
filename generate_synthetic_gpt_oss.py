import asyncio
import multiprocessing
import os
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import aiohttp
from datasets import Dataset, load_dataset
from tqdm.asyncio import tqdm_asyncio
from transformers import HfArgumentParser
import wandb
import yaml

HF_TOKEN = os.environ.get("HF_TOKEN", None)


@dataclass
class GPTOSSSwarmConfig:
    """Configuration for GPT-OSS-120B API swarm"""
    model: str = "gpt-oss-120b"
    api_base: str = "https://your-gpt-oss-120b-api-endpoint.com/v1"
    api_key: str = os.environ.get("API_KEY", "")
    instances: int = 1
    per_instance_max_parallel_requests: int = 10
    request_timeout: int = 120
    max_retries: int = 6
    retry_delay: int = 4


@dataclass
class Args:
    # Generation parameters
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_sequences: List[str] = None
    
    # Prompts dataset parameters
    prompts_dataset: str = "your/prompts/dataset"
    max_samples: int = 1000
    start_sample: int = -1
    end_sample: int = -1
    seed: int = 42
    prompt_column: str = "prompt"
    shuffle_dataset: bool = False
    debug: bool = False
    
    # Logging parameters
    repo_id: str = "your-username/synthetic_data_gpt_oss"
    checkpoint_path: str = "./synthetic_data"
    checkpoint_interval: int = 100
    wandb_username: str = "your-wandb-username"
    min_token_length: int = 150
    push_to_hub: bool = True
    
    # API configuration
    config_file: Optional[str] = None


class GPTOSSClient:
    """Client for GPT-OSS-120B API"""
    
    def __init__(self, config: GPTOSSSwarmConfig):
        self.config = config
        self.session = None
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate_text(self, prompt: str, generation_params: dict) -> dict:
        """Generate text using GPT-OSS-120B API"""
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": generation_params.get("max_new_tokens", 2048),
            "temperature": generation_params.get("temperature", 0.7),
            "top_p": generation_params.get("top_p", 0.9),
            "stop": generation_params.get("stop_sequences", []),
            "stream": False
        }
        
        async with self.session.post(
            f"{self.config.api_base}/chat/completions",
            headers=self.headers,
            json=payload
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result
            else:
                error_text = await response.text()
                raise Exception(f"API error {response.status}: {error_text}")


def load_config(config_file: Optional[str] = None) -> GPTOSSSwarmConfig:
    """Load configuration from file or use defaults"""
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        return GPTOSSSwarmConfig(**config_data)
    return GPTOSSSwarmConfig()


async def process_text(sample, client, generation_params, args, tokenizer, semaphore):
    """Process a single text sample"""
    token_length = 0
    attempt = 0
    
    while attempt < args.max_retries:
        try:
            async with semaphore:
                result = await client.generate_text(
                    sample[args.prompt_column], 
                    generation_params
                )
                
                completion_text = result["choices"][0]["message"]["content"]
                
                # Process stop sequences
                for stop_seq in generation_params.get("stop_sequences", []):
                    if completion_text.endswith(stop_seq):
                        completion_text = completion_text[:-len(stop_seq)].rstrip()
                
                # Estimate token length (adjust based on your tokenizer)
                token_length = len(completion_text.split())  # Simple approximation
                
                sample["completion"] = completion_text
                sample["token_length"] = token_length
                sample["model"] = args.model
                return sample

        except Exception as e:
            attempt += 1
            if attempt < args.max_retries:
                print(f"Request failed, retrying in {args.retry_delay} seconds... (Attempt {attempt}/{args.max_retries})")
                await asyncio.sleep(args.retry_delay)
            else:
                print(f"Max retries reached. Failed to process the request with error {str(e)}.")
                sample["completion"] = ""
                sample["token_length"] = 0
                return sample


async def main():
    """Main async function"""
    parser = HfArgumentParser((Args,))
    args = parser.parse_args_into_dataclasses()[0]
    
    # Load GPT-OSS configuration
    gpt_config = load_config(args.config_file)
    
    # Initialize WandB
    wandb.init(
        project="synthetic_data_gpt_oss",
        entity=args.wandb_username,
        name=args.repo_id.split("/")[1],
    )
    wandb.config.update(asdict(args))
    wandb.config.update(asdict(gpt_config))
    
    # Load prompts dataset
    num_proc = 1 if args.debug else multiprocessing.cpu_count()
    ds = load_dataset(
        args.prompts_dataset, token=HF_TOKEN, split="train", num_proc=num_proc
    )
    
    if args.shuffle_dataset:
        ds = ds.shuffle(seed=args.seed)
    
    if args.start_sample >= 0:
        end_sample = len(ds) if args.end_sample < 0 else args.end_sample
        print(f"Loading a defined range of samples: ({args.start_sample}, {end_sample})...")
        ds = ds.select(range(args.start_sample, end_sample))
    elif args.max_samples > 0:
        print(f"Loading the first {args.max_samples} samples...")
        ds = ds.select(range(args.max_samples))
    
    # Prepare generation parameters
    generation_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "stop_sequences": args.stop_sequences or ["<|endoftext|>", "\n\n\n"]
    }
    
    # Create checkpoint directory
    checkpoint_dir = f"{args.checkpoint_path}/{args.repo_id.split('/')[1]}/data"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Will be saving at {checkpoint_dir}")
    
    # Initialize GPT-OSS client
    async with GPTOSSClient(gpt_config) as client:
        semaphore = asyncio.Semaphore(gpt_config.per_instance_max_parallel_requests)
        
        start_time = time.time()
        total_tokens = 0
        saving_time = 0
        total_samples = len(ds)
        
        for i in range(0, total_samples, args.checkpoint_interval):
            batch_time = time.time()
            print(f"Processing chunk {int(i/args.checkpoint_interval)}/{int(total_samples/args.checkpoint_interval)}")
            
            end_index = min(i + args.checkpoint_interval, total_samples)
            chunk = ds.select(range(i, end_index))
            
            # Process chunk
            chunk_results = await tqdm_asyncio.gather(
                *(process_text(sample, client, generation_params, args, None, semaphore) for sample in chunk)
            )
            
            # Save checkpoint
            temp_time = time.time()
            time_per_chunk = temp_time - batch_time
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{i}.json")
            
            intermediate_ds = Dataset.from_list(chunk_results)
            intermediate_ds.to_json(checkpoint_path)
            
            batch_tokens = sum(intermediate_ds["token_length"])
            total_tokens += batch_tokens
            saving_time += time.time() - temp_time
            
            print(f"💾 Checkpoint (samples {i}-{end_index}) saved at {checkpoint_path}.")
            
            # Log to WandB
            wandb.log({
                "sample": end_index,
                "batch": int(i / args.checkpoint_interval),
                "total_tokens (M)": total_tokens / 1e6,
                "tokens_per_batch": batch_tokens,
                "time_per_batch (s)": time_per_chunk,
                "generated_tokens_per_sec": int(batch_tokens / time_per_chunk),
            })
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        print("Done processing and saving all chunks 🎉!")
        print(f"🏎️💨 Overall Tokens per Second: {total_tokens / total_duration:.2f}")
        print(f"Generated {total_tokens / 1e6:.2f}M tokens")
        print(f"Total duration: {total_duration // 3600}h{int((total_duration % 3600) // 60)}min")
        print(f"Saving time: {saving_time}s={saving_time/60}min")
        
        # Load and filter final dataset
        print("Load checkpoints...")
        output_ds = load_dataset(checkpoint_dir, split="train")
        final_data = output_ds.filter(lambda x: x["token_length"] >= args.min_token_length)
        
        print(final_data)
        
        if args.push_to_hub:
            print(f"📨 Pushing dataset to {args.repo_id}")
            final_data.push_to_hub(args.repo_id, private=True)
            print("Dataset pushed!")
            
            failed = output_ds.filter(lambda x: x["token_length"] < args.min_token_length)
            if len(failed) > 0:
                print(f"{len(failed)} generations failed")
                failed.push_to_hub(f"{args.repo_id}_failed", private=True)


if __name__ == "__main__":
    asyncio.run(main())
    wandb.finish()