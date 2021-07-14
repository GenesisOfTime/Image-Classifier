import matplotlib.pyplot as plt
from PIL import Image
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import json



def application(model_name, category_names, image_path, top_k):
    
    processed_image = process_image(image_path)
    model_name = load_model(model_name)
    prediction = model_name.predict(processed_image)
    probabilities, classes = predict(prediction, model_name, top_k)
    category_names = load_json(category_names)
    
    character_class_names = []
    for i in classes:
        character_class_names.append(category_names[i])
    print('Top probabilities for the given image: {}'.format(probabilities))
    print('Top classes for the given image: {}'.format(character_class_names))


def load_model(model_name):
    reloaded_saved_model = tf.keras.models.load_model(model_name, compile = False, custom_objects = {'KerasLayer': hub.KerasLayer})
    return reloaded_saved_model

def load_json(category_names):
    with open(category_names, 'r') as f:
        category_names = json.load(f)
        return category_names
             

    
def process_image(image_path):
    processed_image = Image.open(image_path)
    processed_image = np.asarray(processed_image)
    processed_image = tf.convert_to_tensor(processed_image, tf.float32) 
    processed_image = tf.image.resize(processed_image, (224, 224)).numpy()/255
    processed_image = np.expand_dims(processed_image, axis = 0)
    
    
    return processed_image



def predict(prediction, model, top_k):  
   
    class_to_prob = {str(class_): prob for class_, prob in enumerate(prediction[0], 1)}
    class_to_prob = sorted(class_to_prob.items(), key = lambda zerotoone: zerotoone[1], reverse = True)
    class_to_prob = dict(class_to_prob)
    class_to_prob_values = list(class_to_prob.values())
    class_to_prob = list(class_to_prob)

    
    return class_to_prob_values[:top_k], class_to_prob[:top_k]
    

parser = argparse.ArgumentParser(description='Flower Image Classifier')
    
parser.add_argument('image_path', type=str, default=None, help='Path for the image')
parser.add_argument('model', type=str, default=None, help='Load the model')
parser.add_argument('--top_k', type=int, default=5, help='Top K probabilities', action = "store")
parser.add_argument('--category_names', type=str, default='label_map.json', help='Path for JSON file that maps names to categories',  action = "store")
args = (parser.parse_args())
  
args.image_path
args.model
args.top_k
args.category_names
if __name__ == '__main__':
    application(args.model, args.category_names, args.image_path, args.top_k) 
    