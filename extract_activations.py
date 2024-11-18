import torch
from transformers import AutoModel, AutoTokenizer

def extract_activations(texts, model_name="distilbert-base-uncased", layers_to_analyze=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    
    activations = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # Tuple of tensors for each layer
            
            if layers_to_analyze:
                selected_layers = [hidden_states[i] for i in layers_to_analyze]
            else:
                selected_layers = hidden_states  # All layers
            
            activations.append(selected_layers)
    
    return activations

if __name__ == "__main__":
    texts = ["I love this movie!", "This is a bad day."]
    activations = extract_activations(texts)
    torch.save(activations, "activations.pt")
    print("Activations saved!")
