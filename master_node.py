from mpi4py import MPI
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import threading

class MasterNode:
    def __init__(self, comm):
        self.comm = comm
        self.size = comm.Get_size()
        self.root = tk.Tk()
        self.root.title("MPI Cluster Control - Lab 05 NLP")
        self.root.geometry("600x400")
        
        self.setup_gui()
        self.dataset = self.load_dataset()

    def load_dataset(self):
        # Synthetic dataset for demonstration purposes
        # In a real scenario, this would load a CSV
        data = {
            'text': [
                "I love this movie", "this is great", "fantastic plot", "best ever", "loved it",
                "terrible movie", "waste of time", "awful", "worst acting", "hated it",
                "so boring", "not good", "very bad", "excellent", "masterpiece",
                "horrible", "disaster", "good job", "well done", "amazing"
            ] * 50, # Duplicate to have enough data
            'label': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1] * 50
        }
        return (data['text'], data['label'])

    def setup_gui(self):
        frame = ttk.Frame(self.root, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Master Node Control Panel", font=("Arial", 14, "bold")).pack(pady=10)
        
        ttk.Button(frame, text="Start Distributed Training", command=self.start_training).pack(pady=10)
        
        self.results_text = tk.Text(frame, height=15, width=60)
        self.results_text.pack(pady=10)

    def log(self, message):
        self.results_text.insert(tk.END, message + "\n")
        self.results_text.see(tk.END)

    def start_training(self):
        self.log("Starting distribution...")
        # Run MPI operations in a separate thread to not freeze GUI
        threading.Thread(target=self.run_mpi_tasks, daemon=True).start()

    def run_mpi_tasks(self):
        # Send data to children
        for i in range(1, self.size):
            self.log(f"Sending dataset to Rank {i}...")
            self.comm.send(self.dataset, dest=i, tag=11)
        
        # Receive results
        for i in range(1, self.size):
            self.log(f"Waiting for results from Rank {i}...")
            result = self.comm.recv(source=i, tag=22)
            self.log(f"Received result from Rank {i}:")
            self.log(f"  Model: {result['model']}")
            self.log(f"  Accuracy: {result['accuracy']:.4f}")
            self.log(f"  Time: {result['time']:.4f}s")
            self.log("-" * 30)
            
        self.log("All tasks completed.")

    def run(self):
        self.root.mainloop()
