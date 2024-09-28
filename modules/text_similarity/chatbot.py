import sys
import pickle
import json
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLineEdit, QTextEdit, QPushButton)
from PyQt5.QtCore import Qt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

with open('./meta/qa_pairs.json', 'r') as json_file:
    qa_pairs = json.load(json_file)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_answer(input_question):
    with open('data/text-sim/question_embeddings.pkl', 'rb') as f:
        questions, question_embeddings = pickle.load(f)
    input_embedding = model.encode([input_question])
    similarities = cosine_similarity(input_embedding, question_embeddings)
    closest_index = np.argmax(similarities)
    return qa_pairs[closest_index-1]['answer']

class ChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AI Powered MediKit - Bot')
        self.setGeometry(100, 100, 1080, 1080)
        self.setStyleSheet("""
            QWidget {
               background-repeat: no-repeat;
               background-position: center;
               background-size: cover;
            }
            * {
               color: lightblue;
            }
            QLineEdit {
               color: white;
               font-size: 18px;
            }
        """)
        self.layout = QVBoxLayout()
        self.chat_area = QTextEdit()
        self.chat_area.setStyleSheet("""
            QTextEdit {
               background-image: url('./assets/logos/chatbot.png');
               background-repeat: no-repeat;
               background-position: center;
               background-attachment: fixed;
               background-size: cover;
               color: lightblue;
               font-size: 16px;
               background-color: rgba(0, 0, 0, 0.3);  /* Adds semi-transparent black overlay */
            }
        """)
        self.chat_area.setReadOnly(True)
        self.input_line = QLineEdit()
        self.input_line.returnPressed.connect(self.send_message)
        self.send_button = QPushButton('SUBMIT')
        self.send_button.clicked.connect(self.send_message)
        self.layout.addWidget(self.chat_area)
        self.layout.addWidget(self.input_line)
        self.layout.addWidget(self.send_button)
        
        self.setLayout(self.layout)
        
    def send_message(self):
        user_input = self.input_line.text().strip()
        if user_input:
            self.chat_area.append(f"<div style='color:white; background-color: rgba(0,0,0,0.3);'>{user_input.title()}</div>")
            self.input_line.clear()
            response = get_answer(user_input)
            self.chat_area.append(f"<div style='color:lightblue; background-color: rgba(0,0,0,0.3);'>{response}</div>")
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

def init():
    app = QApplication(sys.argv)
    chat_app = ChatApp()
    chat_app.show()
    sys.exit(app.exec_())