import gradio as gr
from fastai.vision.all import *

def classify_img(img):
    pred , idx , probs = learn.predict(img)
    return dict(zip(categories , map(float , probs)))

import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

learn = load_learner('bears_model.pkl')

categories = ( 'black bear','grizzly bear'  , 'teddy bear')


image = gr.inputs.Image(shape=(194,194))
label = gr.outputs.Label()
examples = [
    'teddy_bear.jpg' , 
    'grizzly_bear.jpg' ,
    'black_bear.jpg'
]

intf = gr.Interface(fn = classify_img , inputs = image , outputs = label , examples = examples)
intf.launch(inline = False)