from flask import Flask, request, send_file
import io
import torch
import traceback
from torch import autocast
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
app = Flask(__name__)



IS_RUNNING = False

# load sd 
model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
pipe.enable_attention_slicing()
pipe = pipe.to("cuda")

def run_inference(**kwargs):

  kwargs['cfg_scale'] = float(kwargs['cfg_scale'])
  kwargs['steps'] = int(kwargs['steps'])
  kwargs['height'] = int(kwargs['height'])
  kwargs['width'] = int(kwargs['width'])
  kwargs['seed'] = int(kwargs['seed'])
  kwargs['strength'] = float(kwargs['strength'])

  image = pipe(**kwargs)[0][0]
  img_data = io.BytesIO()
  image.save(img_data, "PNG")
  img_data.seek(0)
  return img_data

@app.route('/')
def myapp():
    global IS_RUNNING


    if not IS_RUNNING:
        IS_RUNNING = True
        try:
            img_data = run_inference(**request.args)
        except:
            print(traceback.print_exc())
            IS_RUNNING = False
            return "Servidor deu erro vai ver o que é", 400
    else:
        return "Servidor ocupado", 400
    IS_RUNNING = False
    return send_file(img_data, mimetype='image/png')
