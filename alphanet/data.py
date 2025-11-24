import numpy as np
from datasets import load_from_disk

class AxiomLoader:
    """
    Generates fundamental truths for imprinting based on:
    1. DeepMind Mathematics (Algebra/Arithmetic)
    2. Logic/Reasoning (Transitive properties, Syllogisms)
    3. Knowledge Graphs (RDF Triples like Wikidata)
    """
    def __init__(self):
        self.vocab = sorted(list(set(
            "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "+-*/=><(),.?: "
        )))
        self.char_to_int = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_char = {i: c for i, c in enumerate(self.vocab)}
        
    def generate_math_axiom(self):
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
    """
    Loads Orca Math dataset (200K+ math reasoning examples)
    Format: Question -> Answer pairs
    """
    def __init__(self, dataset_path="./orca_math_local"):
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
        """
        Returns batches of (question + answer) text for character-level modeling
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
    def __init__(self, text_source, vocab_override=None):
        self.text = text_source
        if vocab_override:
            self.vocab = vocab_override
        else:
            self.vocab = sorted(list(set(text_source)))
            
        self.char_to_int = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_char = {i: c for i, c in enumerate(self.vocab)}
        self.ptr = 0
        
    def get_batch(self, batch_size=32, seq_len=10):
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
    def __init__(self, url):
        print(f"YouTubeLoader initialized for: {url}")
    def get_batch(self, batch_size=10):
        return np.random.rand(batch_size, 64*64*3)

class WebcamLoader:
    def get_frame(self):
        return np.random.rand(64*64*3)


class UCF101VideoLoader:
    """
    Real UCF101 dataset loader for action recognition.
    Processes actual video files from the UCF-101 dataset.
    """
    def __init__(self, dataset_dir="./UCF-101", num_classes=10, frame_size=64, num_frames=16):
        """
        Args:
            dataset_dir: Path to UCF-101 directory
            num_classes: Number of action classes to use (max 101)
            frame_size: Size to resize frames to
            num_frames: Number of frames to sample per video
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
        """Load and process a single video file."""
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
        """
        Returns a batch of real video clips and labels.
        
        Returns:
            videos: (batch_size, num_frames, frame_size*frame_size*3)
            labels: (batch_size,) class indices
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
        """Fallback synthetic data generator."""
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
    """Legacy synthetic video loader (kept for backward compatibility)."""
    def __init__(self, video_dir=None, num_classes=10, frame_size=64, num_frames=16):
        self.num_classes = num_classes
        self.frame_size = frame_size
        self.num_frames = num_frames
        print(f"VideoLoader (synthetic) initialized: {num_classes} classes, {frame_size}x{frame_size}, {num_frames} frames")
        
    def get_batch(self, batch_size=8):
        videos = []
        labels = []
        for _ in range(batch_size):
            frames = [np.random.rand(self.frame_size * self.frame_size * 3) for _ in range(self.num_frames)]
            videos.append(np.array(frames))
            labels.append(np.random.randint(0, self.num_classes))
        return np.array(videos), np.array(labels)
    """
    Loads video data for action recognition.
    Processes videos as sequences of frames for temporal modeling.
    """
    def __init__(self, video_dir=None, num_classes=10, frame_size=64, num_frames=16):
        """
        Args:
            video_dir: Path to video directory (optional, uses synthetic data if None)
            num_classes: Number of action classes
            frame_size: Size to resize frames to (frame_size x frame_size)
            num_frames: Number of frames per video clip
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
        """
        Returns a batch of video clips and labels.
        
        Returns:
            videos: (batch_size, num_frames, frame_size*frame_size*3)
            labels: (batch_size,) class indices
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
        """
        Placeholder for loading real video files.
        Use opencv-python (cv2) to load and process videos.
        
        Example implementation:
        ```python
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize and flatten
            frame = cv2.resize(frame, (self.frame_size, self.frame_size))
            frame = frame.flatten() / 255.0  # Normalize
            frames.append(frame)
        cap.release()
        return np.array(frames)
        ```
        """
        pass

