import modal
import os
import uuid
import base64
import requests
from pydantic import BaseModel
app=modal.App("Algo Rhythm")
image=(
    modal.Image.debian_slim()
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(["git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step","cd /tmp/ACE-Step && pip install ."])
    .env({"HF_HOME":"/.cache/huggingface"})
    .add_local_python_source("prompts")
)

model_volume = modal.Volume.from_name(
    "ace-step-models", create_if_missing=True)
hf_volume = modal.Volume.from_name("qwen-hf-cache", create_if_missing=True)

aws_secret=modal.Secret.from_name("algo-rhytm-secret")

class AudioGenerationBase(BaseModel):
    audio_duration: float = 180.0
    seed:int= -1
    guidance_scale: float = 15.0
    infer_step: int = 60

class GenerateFromDescriptionRequest(AudioGenerationBase):
    full_described_song:str

class GenerateWithCustomLyricsRequest(AudioGenerationBase):
    prompt:str
    lyrics:str 

class GenerateWithDescribedLyricsRequest(AudioGenerationBase):
    prompt:str 
    described_lyrics:str      

class GenerateMusicResponseS3(BaseModel):
    s3_key:str
    covere_image_s3_key:str
    categories: list[str]
    audio_data:str


class GenerateMusicResponse(BaseModel):
    audio_data:str
    

@app.cls(
    image=image,
    gpu="L40S",
     volumes={"/models": model_volume, "/.cache/huggingface": hf_volume},
    secrets=[aws_secret],
    scaledown_window=15

)

class MusicGenServer:
    @modal.enter()
    def load_model(self):
        from acestep.pipeline_ace_step import ACEStepPipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from diffusers import AutoPipelineForText2Image
        import torch

        #  The Music  Model
        self.music_model = ACEStepPipeline(
            checkpoint_dir="/models",
            dtype="bfloat16",
            torch_compile=False,
            cpu_offload=False,
            overlapped_decode=False
        )

        # The LLM
        model_id = "Qwen/Qwen2-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="/.cache/huggingface"
        )

        # (thumbnails) The Stable Diffusion Model
        self.image_pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", cache_dir="/.cache/huggingface")
        self.image_pipe.to("cuda")

    def prompt_qwen(self, question: str):
        messages = [{"role": "user", "content": question}]
        text = self.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm_model.device)

        generated_ids = self.llm_model.generate(
        model_inputs.input_ids, max_new_tokens=512
         )
        generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids
        in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


    def generate_prompt(self,description:str):
        #for llm description
        full_prompt=PROMPT_GENERATOR_PROMPT.format(user_prompt=description)

        #run then return 
        return self.prompt_qwen(full_prompt)

    def generate_lyrics(self, description: str):
        # llm to genr the lyrics 
        full_prompt = LYRICS_GENERATOR_PROMPT.format(description=description)

    def generate_and_upload_to_s3(
            self,
            prompt: str,
            lyrics: str,
            instrumental: bool,
            audio_duration: float,
            infer_step: int,
            guidance_scale: float,
            seed: int,
    ) -> GenerateMusicResponseS3:
        final_lyrics = "[instrumental]" if instrumental else lyrics
        print(f"Generated lyrics: \n{final_lyrics}")
        print(f"Prompt: \n{prompt}")

    @modal.fastapi_endpoint(method="POST")
    def generate(self) -> GenerateMusicResponse:
        output_dir = "/tmp/outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")

        self.music_model(
            prompt="electronic rap",
            lyrics="[verse]\nWaves on the bass, pulsing in the speakers,\nTurn the dial up, we chasing six-figure features,\nGrinding on the beats, codes in the creases,\nDigital hustler, midnight in sneakers.\n\n[chorus]\nElectro vibes, hearts beat with the hum,\nUrban legends ride, we ain't ever numb,\nCircuits sparking live, tapping on the drum,\nLiving on the edge, never succumb.\n\n[verse]\nSynthesizers blaze, city lights a glow,\nRhythm in the haze, moving with the flow,\nSwagger on stage, energy to blow,\nFrom the blocks to the booth, you already know.\n\n[bridge]\nNight's electric, streets full of dreams,\nBass hits collective, bursting at seams,\nHustle perspective, all in the schemes,\nRise and reflective, ain't no in-betweens.\n\n[verse]\nVibin' with the crew, sync in the wire,\nGot the dance moves, fire in the attire,\nRhythm and blues, soul's our supplier,\nRun the digital zoo, higher and higher.\n\n[chorus]\nElectro vibes, hearts beat with the hum,\nUrban legends ride, we ain't ever numb,\nCircuits sparking live, tapping on the drum,\nLiving on the edge, never succumb.",
            audio_duration=180,
            infer_step=60,
            guidance_scale=15,
            save_path=output_path,
        )

        with open(output_path, "rb") as f:
            audio_bytes = f.read()

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        os.remove(output_path)

        return GenerateMusicResponse(audio_data=audio_b64)

    
  
    @modal.fastapi_endpoint(method="POST")
    def generate_from_descripiton(self,request:GenerateFromDescriptionRequest) -> GenerateMusicResponse:
        prompt=self.generate_prompt(request.full_described_song)

        #so using description we need to generate prompt to pass to model for tag 
        #also we hv to generate lyrics 

        lyrics=""
        if not request.instrumental:
            lyrics=self.generate_lyrics(request.full_described_song)
        

    @modal.fastapi_endpoint(method="POST")
    def generate_with_described_lyrics(self,request:GenerateWithDescribedLyricsRequest) -> GenerateMusicResponse:
        # in this we still need llm to generate lyrics
        pass

    @modal.fastapi_endpoint(method="POST")
    def generate_with_lyrics(self,request:GenerateWithCustomLyricsRequest) -> GenerateMusicResponse:
        pass




@app.local_entrypoint()
def main():    
    server=MusicGenServer()
    endpoint_url=server.generate.get_web_url()

    response=requests.post(endpoint_url)
    response.raise_for_status()
    result=GenerateMusicResponse(**response.json())

    audio_bytes=base64.b64decode(result.audio_data)
    ouput_filename="generated.wav"
    with open(ouput_filename,"wb") as f:
        f.write(audio_bytes)
