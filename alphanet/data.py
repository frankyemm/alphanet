import numpy as np
from datasets import load_from_disk

class AxiomLoader:
    """Generates fundamental truth axioms for model imprinting.

    This loader generates synthetic data representing different types of fundamental truths:
    algebraic/arithmetic relationships, logical reasoning (transitivity), and knowledge graph triples.

    Attributes:
        vocab (list): List of all allowed characters in the generated axioms.
        char_to_int (dict): Mapping from character to integer index.
        int_to_char (dict): Mapping from integer index to character.
    """
    def __init__(self):
        """Initializes the AxiomLoader with a predefined vocabulary."""
        self.vocab = sorted(list(set(
            "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "+-*/=><(),.?: "
        )))
        self.char_to_int = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_char = {i: c for i, c in enumerate(self.vocab)}
        
    def generate_math_axiom(self):
        """Generates a mathematical axiom (DeepMind Math Style).

        Examples:
            "5+3=8"
            "7-2=5"
            "3*4=12"

        Returns:
            str: A string representing a simple arithmetic equation.
        """
        """DeepMind Math Style: Algebra & Arithmetic"""
        op = np.random.choice(['+', '-', '*'])
        a = np.random.randint(1, 20)
        b = np.random.randint(1, 20)
        
        if op == '+':
            res = a + b
            return f"{a}+{b}={res}"
        elif op == '-':
            # Ensure positive for simplicity in this demo
            if a < b: a, b = b, a
            res = a - b
            return f"{a}-{b}={res}"
        elif op == '*':
            # Keep numbers small for multiplication
            a = np.random.randint(1, 10)
            b = np.random.randint(1, 10)
            res = a * b
            return f"{a}*{b}={res}"
            
    def generate_logic_axiom(self):
        """Generates a logical axiom involving transitive relations.

        Examples:
            "A>B & B>C => A>C"
            "If A->B & B->C Then A->C"

        Returns:
            str: A string representing a logical implication or order relation.
        """
        """Logic/CLUTRR Style: Transitive Relations"""
        # A > B, B > C => A > C
        entities = list("ABCDE")
        np.random.shuffle(entities)
        x, y, z = entities[:3]
        
        types = ['>', '<', '->']
        rel = np.random.choice(types)
        
        if rel == '->': # Implication
            return f"If {x}->{y} & {y}->{z} Then {x}->{z}"
        else: # Order
            return f"{x}{rel}{y} & {y}{rel}{z} => {x}{rel}{z}"
            
    def generate_fact_axiom(self):
        """Generates a factual axiom (RDF Triple style).

        Examples:
            "(Paris, capital_of, France)"
            "(Water, formula, H2O)"

        Returns:
            str: A string representing a fact in triple format.
        """
        """Wikidata/ConceptNet Style: RDF Triples"""
        facts = [
            "(Water, formula, H2O)",
            "(Paris, capital_of, France)",
            "(Earth, orbits, Sun)",
            "(Triangle, sides, 3)",
            "(Square, sides, 4)",
            "(Human, instance_of, Mammal)",
            "(Fire, produces, Heat)",
            "(Ice, state, Solid)",
            "(Rain, state, Liquid)"
        ]
        return np.random.choice(facts)

    def get_batch(self, batch_size=32, seq_len=15):
        """Generates a batch of synthetic axiom data.

        Args:
            batch_size (int): Number of samples in the batch. Defaults to 32.
            seq_len (int): Length of the input sequence. Defaults to 15.

        Returns:
            tuple: A tuple containing:
                - inputs (numpy.ndarray): Input tensor of shape (Batch, Time, 1), normalized to [0, 1].
                - targets (numpy.ndarray): Target class indices of shape (Batch,), representing the next character.
        """
        inputs = []
        targets = []
        
        for _ in range(batch_size):
            # Mix different types of truths
            r = np.random.rand()
            if r < 0.4:
                axiom = self.generate_math_axiom()
            elif r < 0.7:
                axiom = self.generate_logic_axiom()
            else:
                axiom = self.generate_fact_axiom()
                
            # Pad or crop
            if len(axiom) < seq_len + 1:
                axiom = axiom + " " * (seq_len + 1 - len(axiom))
            
            # Sliding window (simplified to start)
            chunk = axiom[0 : seq_len]
            target_char = axiom[seq_len]
            
            x = [self.char_to_int.get(c, 0) / len(self.vocab) for c in chunk]
            y = self.char_to_int.get(target_char, 0)
            
            inputs.append(x)
            targets.append(y)
            
        # Reshape inputs to (Batch, Time, 1) for compatibility with QBitSNN
        return np.array(inputs)[..., np.newaxis], np.array(targets)


class OrcaMathLoader:
    """Loader for the Orca Math dataset.

    This loader handles loading and processing of math reasoning examples (Question -> Answer pairs).
    It builds a vocabulary from the dataset and provides character-level batches.

    Attributes:
        dataset (datasets.Dataset): The loaded dataset object.
        vocab (list): List of unique characters found in the dataset sample.
        char_to_int (dict): Mapping from character to integer index.
        int_to_char (dict): Mapping from integer index to character.
        current_idx (int): Current index pointer in the dataset for sequential access.
    """
    def __init__(self, dataset_path="./orca_math_local"):
        """Initializes the OrcaMathLoader.

        Args:
            dataset_path (str): Path to the local dataset directory. Defaults to "./orca_math_local".
        """
        print(f"Loading Orca Math dataset from {dataset_path}...")
        self.dataset = load_from_disk(dataset_path)
        print(f"Loaded {len(self.dataset)} examples")
        
        # Build vocabulary from the dataset
        print("Building vocabulary...")
        all_text = ""
        for i in range(min(10000, len(self.dataset))):  # Sample 10k for vocab
            all_text += self.dataset[i]['question'] + " " + self.dataset[i]['answer'] + " "
        
        self.vocab = sorted(list(set(all_text)))
        self.char_to_int = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_char = {i: c for i, c in enumerate(self.vocab)}
        print(f"Vocabulary size: {len(self.vocab)} characters")
        
        self.current_idx = 0
        
    def get_batch(self, batch_size=32, seq_len=128):
        """Returns a batch of character-level examples from the Orca Math dataset.

        Each example is formatted as "Q: {question} A: {answer}".

        Args:
            batch_size (int): Number of examples per batch. Defaults to 32.
            seq_len (int): Length of the input sequence. Defaults to 128.

        Returns:
            tuple: A tuple containing:
                - inputs (numpy.ndarray): Input tensor of shape (Batch, Time, 1), normalized to [0, 1].
                - targets (numpy.ndarray): Target class indices of shape (Batch,), representing the next character.
        """
        inputs = []
        targets = []
        
        for _ in range(batch_size):
            # Get next example (cycle through dataset)
            if self.current_idx >= len(self.dataset):
                self.current_idx = 0
                
            example = self.dataset[self.current_idx]
            # Combine question and answer with separator
            text = f"Q: {example['question']} A: {example['answer']}"
            
            # Filter out characters not in vocab
            text = "".join([c for c in text if c in self.vocab])
            
            # Ensure we have enough text
            if len(text) < seq_len + 1:
                text = text + " " * (seq_len + 1 - len(text))
            
            # Take a random slice if text is too long
            if len(text) > seq_len + 1:
                start = np.random.randint(0, len(text) - seq_len - 1)
                text = text[start:start + seq_len + 1]
            
            # Create input/target pairs
            chunk = text[:seq_len]
            target_char = text[seq_len]
            
            x = [self.char_to_int.get(c, 0) / len(self.vocab) for c in chunk]
            y = self.char_to_int.get(target_char, 0)
            
            inputs.append(x)
            targets.append(y)
            
            self.current_idx += 1
            
        # Reshape inputs to (Batch, Time, 1)
        return np.array(inputs)[..., np.newaxis], np.array(targets)


class TextLoader:
    """Generic text loader for character-level modeling.

    Attributes:
        text (str): The source text data.
        vocab (list): List of unique characters in the text (or provided override).
        char_to_int (dict): Mapping from character to integer index.
        int_to_char (dict): Mapping from integer index to character.
        ptr (int): Current pointer position in the text.
    """
    def __init__(self, text_source, vocab_override=None):
        """Initializes the TextLoader.

        Args:
            text_source (str): The raw text string to load.
            vocab_override (list, optional): A list of characters to use as the vocabulary.
                If None, the vocabulary is inferred from the text_source. Defaults to None.
        """
        self.text = text_source
        if vocab_override:
            self.vocab = vocab_override
        else:
            self.vocab = sorted(list(set(text_source)))
            
        self.char_to_int = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_char = {i: c for i, c in enumerate(self.vocab)}
        self.ptr = 0
        
    def get_batch(self, batch_size=32, seq_len=10):
        """Returns a batch of text sequences.

        Args:
            batch_size (int): Number of sequences per batch. Defaults to 32.
            seq_len (int): Length of each sequence. Defaults to 10.

        Returns:
            tuple: A tuple containing:
                - inputs (numpy.ndarray): Input tensor of shape (Batch, Time, 1), normalized to [0, 1].
                - targets (numpy.ndarray): Target indices of shape (Batch, Time) for sequence prediction.
        """
        inputs = []
        targets = []
        for _ in range(batch_size):
            if self.ptr + seq_len + 1 >= len(self.text):
                self.ptr = 0
                
            chunk = self.text[self.ptr : self.ptr + seq_len]
            target_chunk = self.text[self.ptr + 1 : self.ptr + seq_len + 1]
            
            x = [self.char_to_int.get(c, 0) / len(self.vocab) for c in chunk]
            y = [self.char_to_int.get(c, 0) for c in target_chunk]
            
            inputs.append(x)
            targets.append(y)
            self.ptr += seq_len
            
        # Reshape inputs to (Batch, Time, 1)
        return np.array(inputs)[..., np.newaxis], np.array(targets)

class YouTubeLoader:
    """Placeholder loader for YouTube video data.

    This class is intended to interface with YouTube videos but currently returns random data.
    """
    def __init__(self, url):
        """Initializes the YouTubeLoader.

        Args:
            url (str): The URL of the YouTube video.
        """
        print(f"YouTubeLoader initialized for: {url}")

    def get_batch(self, batch_size=10):
        """Returns a batch of random video frames.

        Args:
            batch_size (int): Number of frames. Defaults to 10.

        Returns:
            numpy.ndarray: Random array of shape (batch_size, 64*64*3).
        """
        return np.random.rand(batch_size, 64*64*3)

class WebcamLoader:
    """Placeholder loader for webcam feed.

    This class is intended to capture frames from a webcam but currently returns random data.
    """
    def get_frame(self):
        """Captures a single frame.

        Returns:
            numpy.ndarray: Random array of shape (64*64*3).
        """
        return np.random.rand(64*64*3)


class UCF101VideoLoader:
    """Real UCF101 dataset loader for action recognition.

    Processes actual video files from the UCF-101 dataset, handling loading,
    resizing, and batching of video clips.

    Attributes:
        dataset_dir (str): Directory containing the UCF-101 dataset.
        num_classes (int): Number of action classes to use.
        frame_size (int): Height/Width to resize frames to.
        num_frames (int): Number of frames to sample per video clip.
        classes (list): List of class names being used.
        class_to_idx (dict): Mapping from class name to index.
        video_paths (list): List of file paths to all video files.
        labels (list): List of label indices corresponding to video_paths.
        current_idx (int): Current index pointer for iteration.
    """
    def __init__(self, dataset_dir="./UCF-101", num_classes=10, frame_size=64, num_frames=16):
        """Initializes the UCF101VideoLoader.

        Args:
            dataset_dir (str): Path to UCF-101 directory. Defaults to "./UCF-101".
            num_classes (int): Number of action classes to use (max 101). Defaults to 10.
            frame_size (int): Size to resize frames to (square). Defaults to 64.
            num_frames (int): Number of frames to sample per video. Defaults to 16.
        """
        import os
        import glob
        
        self.dataset_dir = dataset_dir
        self.num_classes = min(num_classes, 101)
        self.frame_size = frame_size
        self.num_frames = num_frames
        
        # Get all class directories
        all_classes = sorted([d for d in os.listdir(dataset_dir) 
                            if os.path.isdir(os.path.join(dataset_dir, d))])
        
        # Use only the first num_classes
        self.classes = all_classes[:self.num_classes]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all video paths
        self.video_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(dataset_dir, class_name)
            videos = glob.glob(os.path.join(class_dir, "*.avi"))
            
            for video_path in videos:
                self.video_paths.append(video_path)
                self.labels.append(self.class_to_idx[class_name])
        
        self.current_idx = 0
        
        print(f"UCF101VideoLoader initialized:")
        print(f"  - Classes: {self.num_classes} ({', '.join(self.classes[:5])}...)")
        print(f"  - Total videos: {len(self.video_paths)}")
        print(f"  - Frame size: {frame_size}x{frame_size}")
        print(f"  - Frames per clip: {num_frames}")
        
    def load_video(self, video_path):
        """Loads and processes a single video file.

        Args:
            video_path (str): Path to the video file.

        Returns:
            numpy.ndarray or None: Array of shape (num_frames, frame_size*frame_size*3) if successful,
            None if the video is too short or cannot be read.
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get total frames in video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames < self.num_frames:
            cap.release()
            return None  # Skip videos that are too short
        
        # Sample frames uniformly
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                break
                
            # Resize and normalize
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame = frame.flatten().astype(np.float32) / 255.0  # Normalize to [0, 1]
            frames.append(frame)
        
        cap.release()
        
        if len(frames) != self.num_frames:
            return None
            
        return np.array(frames)
    
    def get_batch(self, batch_size=8):
        """Returns a batch of real video clips and labels.
        
        If real videos fail to load, it falls back to synthetic data after multiple attempts.

        Args:
            batch_size (int): Number of videos in the batch. Defaults to 8.

        Returns:
            tuple: A tuple containing:
                - videos (numpy.ndarray): Video data of shape (batch_size, num_frames, frame_size*frame_size*3).
                - labels (numpy.ndarray): Class indices of shape (batch_size,).
        """
        videos = []
        labels = []
        
        attempts = 0
        max_attempts = batch_size * 3  # Try up to 3x batch_size videos
        
        while len(videos) < batch_size and attempts < max_attempts:
            # Cycle through dataset
            if self.current_idx >= len(self.video_paths):
                self.current_idx = 0
                
            video_path = self.video_paths[self.current_idx]
            label = self.labels[self.current_idx]
            
            # Load video
            video = self.load_video(video_path)
            
            if video is not None:
                videos.append(video)
                labels.append(label)
            
            self.current_idx += 1
            attempts += 1
        
        if len(videos) == 0:
            # Fallback to synthetic data if all videos failed
            print("Warning: All videos failed to load, using synthetic data")
            return self._get_synthetic_batch(batch_size)
        
        # Pad if we didn't get enough videos
        while len(videos) < batch_size:
            videos.append(videos[0])  # Duplicate first video
            labels.append(labels[0])
        
        return np.array(videos), np.array(labels)
    
    def _get_synthetic_batch(self, batch_size):
        """Generates a batch of synthetic video data.

        Used as a fallback when real video loading fails.

        Args:
            batch_size (int): Number of synthetic samples to generate.

        Returns:
            tuple: A tuple containing:
                - videos (numpy.ndarray): Random video data.
                - labels (numpy.ndarray): Random class labels.
        """
        videos = []
        labels = []
        for _ in range(batch_size):
            frames = []
            for _ in range(self.num_frames):
                frame = np.random.rand(self.frame_size * self.frame_size * 3).astype(np.float32)
                frames.append(frame)
            videos.append(np.array(frames))
            labels.append(np.random.randint(0, self.num_classes))
        return np.array(videos), np.array(labels)


class VideoLoader:
    """Synthetic video loader for testing and compatibility.

    This class simulates video data generation without requiring actual video files.

    Attributes:
        video_dir (str or None): Path to video directory (unused in synthetic mode).
        num_classes (int): Number of action classes.
        frame_size (int): Size to resize frames to.
        num_frames (int): Number of frames per video clip.
    """
    def __init__(self, video_dir=None, num_classes=10, frame_size=64, num_frames=16):
        """Initializes the VideoLoader.

        Args:
            video_dir (str, optional): Path to video directory. Defaults to None.
            num_classes (int): Number of action classes. Defaults to 10.
            frame_size (int): Size to resize frames to. Defaults to 64.
            num_frames (int): Number of frames per clip. Defaults to 16.
        """
        self.video_dir = video_dir
        self.num_classes = num_classes
        self.frame_size = frame_size
        self.num_frames = num_frames
        
        # For demo: generate synthetic video data
        # In production, replace with actual video loading (cv2.VideoCapture, etc.)
        print(f"VideoLoader initialized:")
        print(f"  - Classes: {num_classes}")
        print(f"  - Frame size: {frame_size}x{frame_size}")
        print(f"  - Frames per clip: {num_frames}")
        
    def get_batch(self, batch_size=8):
        """Returns a batch of synthetic video clips and labels.

        Args:
            batch_size (int): Number of samples per batch. Defaults to 8.

        Returns:
            tuple: A tuple containing:
                - videos (numpy.ndarray): Video data of shape (batch_size, num_frames, frame_size*frame_size*3).
                - labels (numpy.ndarray): Class indices of shape (batch_size,).
        """
        videos = []
        labels = []
        
        for _ in range(batch_size):
            # Generate synthetic video (random frames)
            # In production: load actual video frames
            frames = []
            for _ in range(self.num_frames):
                # Simulate a frame (grayscale or RGB flattened)
                frame = np.random.rand(self.frame_size * self.frame_size * 3)
                frames.append(frame)
            
            video = np.array(frames)  # Shape: (num_frames, features)
            label = np.random.randint(0, self.num_classes)
            
            videos.append(video)
            labels.append(label)
            
        return np.array(videos), np.array(labels)
    
    def load_real_video(self, video_path):
        """Placeholder for loading real video files.
        
        Args:
            video_path (str): Path to the video file.

        Returns:
            None
        """
        pass
