import asyncio
import multiprocessing
import os
import time
import json
from dataclasses import asdict, dataclass
from typing import List, Optional

from datasets import Dataset, load_dataset
from tqdm.asyncio import tqdm_asyncio
from transformers import HfArgumentParser
import wandb
import yaml

# Import OpenAI library
from openai import OpenAI

HF_TOKEN = os.environ.get("HF_TOKEN", None)


@dataclass
class GPTOSSSwarmConfig:
    """Configuration for GPT-OSS-120B API swarm"""
    model: str = "gpt-oss-120b"
    api_base: str = "http://10.211.37.7:9021"  # Update with your server host
    api_key: str = "EMPTY"  # API key is not needed when using base_url
    instances: int = 1
    per_instance_max_parallel_requests: int = 10
    request_timeout: int = 120
    max_retries: int = 6
    retry_delay: int = 4
    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: int = 32000
    batch_size: int = 5  # Reduced batch size for better compatibility
    use_batch_processing: bool = True  # Flag to enable/disable batch processing


@dataclass
class Args:
    # Generation parameters
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

    json_input_path: Optional[str] = None
    json_field: Optional[str] = None


class GPTOSSClient:
    """Client for GPT-OSS-120B API using OpenAI library"""
    
    def __init__(self, config: GPTOSSSwarmConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=f"{config.api_base}/v1",
        )
        self.use_batch = config.use_batch_processing
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def generate_text_single(self, prompt: str) -> dict:
        """Generate text for a single prompt"""
        try:
            print(f"=== DEBUG: Processing single prompt ===")
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                seed=0,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                extra_body={"reasoning": {"effort": "low"}},
                metadata={"output_format": "reasoning_and_final"}
            )
            
            content = response.choices[0].message.content
            print(f"=== DEBUG: Single response length: {len(content)} ===")
            return {"choices": [{"message": {"content": content}}]}
            
        except Exception as e:
            print(f"=== ERROR in generate_text_single: {e} ===")
            import traceback
            traceback.print_exc()
            raise e
    
    async def generate_text_batch(self, prompts: List[str]) -> List[dict]:
        """Generate text using true batch processing with multiple completions"""
        print(f"=== DEBUG: Batch processing {len(prompts)} prompts ===")
        
        try:
            # Create a conversation with multiple user messages for batch processing
            messages = [{"role": "user", "content": prompt} for prompt in prompts]
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                seed=0,
                top_p=self.config.top_p,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                extra_body={"reasoning": {"effort": "low"}},
                metadata={"output_format": "reasoning_and_final"},
                n=len(prompts)  # Request multiple completions
            )
            
            print(f"=== DEBUG: Batch response received with {len(response.choices)} choices ===")
            
            # Map responses to prompts
            results = []
            for i, choice in enumerate(response.choices):
                content = choice.message.content
                print(f"=== DEBUG: Batch response {i} length: {len(content)} ===")
                results.append({"choices": [{"message": {"content": content}}]})
            
            return results
            
        except Exception as e:
            print(f"=== ERROR in generate_text_batch: {e} ===")
            print("=== Falling back to individual requests ===")
            # Fall back to individual requests if batch fails
            return await self.fallback_individual_requests(prompts)
    
    async def fallback_individual_requests(self, prompts: List[str]) -> List[dict]:
        """Fallback to individual requests when batch processing fails"""
        results = []
        for prompt in prompts:
            try:
                result = await self.generate_text_single(prompt)
                results.append(result)
            except Exception as e:
                print(f"=== ERROR in fallback for prompt: {e} ===")
                # Create empty result for failed prompt
                results.append({"choices": [{"message": {"content": ""}}]})
        return results
    
    async def generate_text(self, prompts: List[str]) -> List[dict]:
        """Main generation method that chooses between batch and individual processing"""
        if self.use_batch and len(prompts) > 1:
            return await self.generate_text_batch(prompts)
        else:
            # Process individually
            results = []
            for prompt in prompts:
                result = await self.generate_text_single(prompt)
                results.append(result)
            return results


def load_config(config_file: Optional[str] = None) -> GPTOSSSwarmConfig:
    """Load configuration from file or use defaults"""
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        return GPTOSSSwarmConfig(**config_data)
    return GPTOSSSwarmConfig()


async def process_batch(samples, client, args, semaphore, gpt_config):
    """Process a batch of text samples"""
    prompts = [sample[args.prompt_column] for sample in samples]
    attempt = 0
    
    while attempt < gpt_config.max_retries:
        try:
            async with semaphore:
                results = await client.generate_text(prompts)
                
                # Process each sample with its corresponding response
                processed_samples = []
                for i, (sample, result) in enumerate(zip(samples, results)):
                    completion_text = result["choices"][0]["message"]["content"]
                    
                    # Process stop sequences
                    for stop_seq in args.stop_sequences or []:
                        if completion_text.endswith(stop_seq):
                            completion_text = completion_text[:-len(stop_seq)].rstrip()
                    
                    # Estimate token length
                    token_length = len(completion_text.split())
                    
                    # Create filtered output with only specified fields
                    filtered_sample = {
                        "id": sample.get("id", ""),
                        "category": sample.get("category", ""),
                        "section": sample.get("section", ""),
                        "unit": sample.get("unit", ""),
                        "completion": completion_text,
                        "token_length": token_length,
                        "model": gpt_config.model
                    }
                    
                    processed_samples.append(filtered_sample)
                
                return processed_samples

        except Exception as e:
            attempt += 1
            if attempt < gpt_config.max_retries:
                print(f"Request failed, retrying in {gpt_config.retry_delay} seconds... (Attempt {attempt}/{gpt_config.max_retries})")
                await asyncio.sleep(gpt_config.retry_delay)
            else:
                print(f"Max retries reached. Failed to process the request with error {str(e)}.")
                # Return failed samples with minimal fields
                failed_samples = []
                for sample in samples:
                    failed_sample = {
                        "id": sample.get("id", ""),
                        "category": sample.get("category", ""),
                        "section": sample.get("section", ""),
                        "unit": sample.get("unit", ""),
                        "completion": "",
                        "token_length": 0,
                        "model": gpt_config.model
                    }
                    failed_samples.append(failed_sample)
                return failed_samples


async def main():
    """Main async function"""
    parser = HfArgumentParser((Args,))
    args = parser.parse_args_into_dataclasses()[0]
    
    # Load GPT-OSS configuration
    gpt_config = load_config(args.config_file)
    
    # Load prompts dataset
    num_proc = 1 if args.debug else multiprocessing.cpu_count()
    
    if args.json_input_path:
        ds = load_dataset("json", data_files=args.json_input_path, field=args.json_field, split='train')
    else:
        ds = load_dataset(args.prompts_dataset, token=HF_TOKEN, split='train', num_proc=num_proc)
    
    if args.shuffle_dataset:
        ds = ds.shuffle(seed=args.seed)
    
    if args.start_sample >= 0:
        end_sample = len(ds) if args.end_sample < 0 else args.end_sample
        print(f"Loading a defined range of samples: ({args.start_sample}, {end_sample})...")
        ds = ds.select(range(args.start_sample, end_sample))
    elif args.max_samples > 0:
        print(f"Loading the first {args.max_samples} samples...")
        ds = ds.select(range(args.max_samples))
    
    # Create checkpoint directory
    checkpoint_dir = f"{args.checkpoint_path}/data"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Will be saving at {checkpoint_dir}")
    print(f"Using batch processing: {gpt_config.use_batch_processing}")
    print(f"Batch size: {gpt_config.batch_size}")
    
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
            
            # Process chunk in batches
            chunk_results = []
            batch_size = gpt_config.batch_size
            
            for j in range(0, len(chunk), batch_size):
                batch_end = min(j + batch_size, len(chunk))
                batch = chunk.select(range(j, batch_end))
                print(f"  Processing batch {j//batch_size + 1}/{(len(chunk) + batch_size - 1)//batch_size} with {len(batch)} samples")
                
                batch_results = await process_batch(batch, client, args, semaphore, gpt_config)
                chunk_results.extend(batch_results)
                
                # Log progress for this batch
                successful = sum(1 for r in batch_results if r["token_length"] > 0)
                print(f"  ✅ Batch completed: {successful}/{len(batch)} successful generations")
            
            # Save checkpoint (UTF-8 NDJSON with unescaped Unicode)
            temp_time = time.time()
            time_per_chunk = temp_time - batch_time
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{i}.json")
            
            # Write as JSON Lines to preserve Unicode readability
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                for rec in chunk_results:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
            # Compute tokens for this batch
            batch_tokens = sum(rec.get("token_length", 0) for rec in chunk_results)
            total_tokens += batch_tokens
            saving_time += time.time() - temp_time
            
            successful_in_chunk = sum(1 for r in chunk_results if r["token_length"] > 0)
            print(f"💾 Checkpoint (samples {i}-{end_index}) saved at {checkpoint_path}.")
            print(f"📊 Chunk stats: {successful_in_chunk}/{len(chunk_results)} successful, {batch_tokens} tokens")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        print("Done processing and saving all chunks 🎉!")
        print(f"🏎️💨 Overall Tokens per Second: {total_tokens / total_duration:.2f}")
        print(f"Generated {total_tokens / 1e6:.2f}M tokens")
        print(f"Total duration: {total_duration // 3600}h{int((total_duration % 3600) // 60)}min")
        print(f"Saving time: {saving_time}s={saving_time/60}min")
        
        # Load and filter final dataset from saved NDJSON checkpoints
        print("Load checkpoints...")
        checkpoint_files = [
            os.path.join(checkpoint_dir, name)
            for name in os.listdir(checkpoint_dir)
            if name.startswith("checkpoint_") and name.endswith(".json")
        ]
        checkpoint_files.sort()
        output_ds = load_dataset("json", data_files=checkpoint_files, split="train")
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