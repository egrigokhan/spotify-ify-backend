from flask import Flask
import numpy as np
# from sentence_transformers import SentenceTransformer

app = Flask(__name__)

'''
@app.before_first_request                                                    
def do_heavy_work():                                                        
  track_vectors = np.load('static/track_vectors_33K.npy', allow_pickle='TRUE').item()
  model = SentenceTransformer('paraphrase-MiniLM-L6-v2').eval()
'''

@app.route("/")
def index():
    return "Hello World!"
  
@app.route("/heartbeat")
def heartbeat():
    return {
        "track_vectors" : len(track_vectors),
        "model"         : model 
    }
  
@app.route('/get-tracks-for-text', methods=['POST'])
def get_tracks_for_text():
    request_data = request.get_json()

    text_sentences = request_data['sentences']
    
    uris = []
    current_playlist = []

    for s in text_sentences:
      sen_vec = model.encode([s])[0]

      sim_array = []

      for tv in track_vectors.values():
        sim = cosine(sen_vec, tv)
        sim_array.append(sim)

      sim_array = np.array(sim_array)
      sim_array_ = np.array(sim_array)
      max_index = np.argmax(sim_array)
      max_index_ = np.argmax(sim_array)

      while (max_index in current_playlist):
        sim_array_ = np.delete(sim_array_, max_index_)
        max_index_ = np.where(sim_array_==sim_array_[np.argmax(sim_array_)])[0][0]
        max_index = np.where(sim_array_==sim_array_[max_index_])[0][0]

      current_playlist.append(max_index)
      uris.append(s)

    return uris                                                 

if __name__ == '__main__':                                                  
    app.run(debug=True)
