import sys
from pathlib import Path
from multiprocessing import Process, Queue, cpu_count
from tqdm import tqdm
import torch
import click
import torchaudio

class BaseWorker:
    """
    Base Worker class to handle common processing tasks.

    Subclasses should implement the following methods:
        - load_model(): Load and return the model.
        - process_audio(waveform, sr, **kwargs): Process the waveform and return the desired output.
        - save_output(output, output_path): Save the processed output to the specified path.
    """

    def __init__(self, data_dir, model_path, queue, verbose, overwrite, device_id, **kwargs):
        self.data_dir = data_dir
        self.model_path = model_path
        self.queue = queue
        self.verbose = verbose
        self.overwrite = overwrite
        self.kwargs = kwargs
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()

    def load_model(self):
        """
        Load and return the model.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the load_model method.")

    def process_audio(self, waveform, sr, **kwargs):
        """
        Process the waveform and return the desired output.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the process_audio method.")

    def save_output(self, output, output_path):
        """
        Save the processed output to the specified path.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the save_output method.")

    def handle_audio(self, audio):
        """
        Handle the processing of a single audio entry.
        """
        speaker = audio.get('speaker')
        file_name = audio.get('file_name')

        if not speaker or not file_name:
            if self.verbose:
                self.queue.put(f"Skipping invalid entry: {audio}")
            return

        wav_path = Path(self.data_dir) / speaker / f"{file_name}.wav"
        output_path = self.get_output_path(speaker, file_name)

        if output_path.is_file() and not self.overwrite:
            if self.verbose:
                self.queue.put(f"Skipping existing file: {output_path}")
            self.queue.put("PROGRESS")
            return

        if not wav_path.is_file():
            if self.verbose:
                self.queue.put(f"Warning: WAV file not found: {wav_path}")
            return

        try:
            # Load audio
            waveform, sr = torchaudio.load(str(wav_path))
            waveform = waveform.to(self.device)

            # Ensure waveform has proper shape
            if len(waveform.shape) == 1:
                waveform = waveform.unsqueeze(0)
            elif len(waveform.shape) == 2 and waveform.shape[0] != 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Process audio
            output = self.process_audio(waveform, sr, **self.kwargs)

            # Save output
            self.save_output(output, output_path)

            if self.verbose:
                self.queue.put(f"Saved output: {output_path}")

            # Update progress
            self.queue.put("PROGRESS")

        except Exception as e:
            self.queue.put(f"Error processing {wav_path}: {e}")

    def get_output_path(self, speaker, file_name):
        """
        Determine the output path for the processed file.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the get_output_path method.")

    def run(self, audio_subset):
        """
        Process a subset of audio entries.
        """
        for audio in audio_subset:
            self.handle_audio(audio)
        self.queue.put("COMPLETED")

def run_multiprocessing(
    worker_cls,
    all_audios,
    data_dir,
    model_path,
    num_workers_per_device,
    verbose,
    overwrite=False,
    **kwargs
):
    """
    Generic multiprocessing runner using BaseWorker subclasses.

    Args:
        worker_cls (BaseWorker): The worker class to instantiate.
        all_audios (list): List of all audio entries to process.
        data_dir (Path): Root directory of the preprocessed dataset.
        model_path (str): Path to the pre-trained model.
        num_workers_per_device (int): Number of workers per CUDA device.
        verbose (bool): If True, enable verbose output.
        overwrite (bool): If True, overwrite existing output files.
        **kwargs: Additional keyword arguments for the worker.
    """
    # Detect number of available CUDA devices
    num_devices = torch.cuda.device_count()
    if num_devices > 1:
        devices = list(range(num_devices))
    elif num_devices == 1:
        devices = [0]
    else:
        devices = []

    if verbose:
        click.echo(f"Number of CUDA devices available: {num_devices}")

    if devices:
        click.echo(f"Using CUDA devices: {devices}")
    else:
        click.echo("No CUDA devices available. Using CPU.")

    # Determine total number of workers
    if devices:
        total_workers = num_devices * num_workers_per_device
    else:
        total_workers = num_workers_per_device

    # Adjust number of workers if it exceeds CPU count
    available_cpus = cpu_count()
    if total_workers > available_cpus:
        click.echo(f"Adjusting total workers from {total_workers} to {available_cpus} due to CPU count limitations.")
        total_workers = available_cpus

    if verbose:
        click.echo(f"Total workers: {total_workers}")

    # Split audios among workers
    split_audios = [[] for _ in range(total_workers)]
    for i, audio in enumerate(all_audios):
        split_audios[i % total_workers].append(audio)

    # Create a multiprocessing Queue for communication
    queue = Queue()

    # Create and start processes
    processes = []
    for i in range(total_workers):
        p = Process(
            target=process_wrapper,
            args=(
                worker_cls,          # Pass the class itself, not an instance
                data_dir,
                model_path,
                queue,
                verbose,
                overwrite,
                devices[i % len(devices)] if devices else -1,  # Handle case when no devices
                split_audios[i],
            ),
            kwargs=kwargs,  # Pass the additional keyword arguments here
            name=f"Process-{i}"
        )
        p.start()
        processes.append(p)

    # Initialize tqdm progress bar
    with tqdm(total=len(all_audios), desc="Processing", unit="file") as pbar:
        completed_workers = 0
        while completed_workers < total_workers:
            message = queue.get()
            if message == "PROGRESS":
                pbar.update(1)
            elif isinstance(message, str) and message.startswith("Saved") and verbose:
                pbar.set_postfix({"Last Saved": message})
            elif isinstance(message, str) and message.startswith("Warning") and verbose:
                pbar.write(message)
            elif isinstance(message, str) and message.startswith("Error"):
                pbar.write(message)
            elif isinstance(message, str) and message.startswith("COMPLETED"):
                completed_workers += 1
                if verbose:
                    pbar.write(f"{message}")
            else:
                if verbose:
                    pbar.write(message)

    # Ensure all processes have finished
    for p in processes:
        p.join()

    click.echo("Processing complete.")

def process_wrapper(worker_cls, data_dir, model_path, queue, verbose, overwrite, device_id, audio_subset, **kwargs):
    """
    Wrap the worker's run method to be compatible with multiprocessing.Process.
    Instantiates the worker within the child process.

    Args:
        worker_cls (BaseWorker): The worker class to instantiate.
        data_dir (str): Root directory of the preprocessed dataset.
        model_path (str): Path to the pre-trained model.
        queue (Queue): Multiprocessing queue for communication.
        verbose (bool): If True, enable verbose output.
        overwrite (bool): If True, overwrite existing output files.
        audio_subset (list): A subset of audio entries to process.
        **kwargs: Additional keyword arguments for the worker.
    """
    worker = worker_cls(
        data_dir=data_dir,
        model_path=model_path,
        queue=queue,
        verbose=verbose,
        overwrite=overwrite,
        device_id=device_id,
        **kwargs  # Forward the additional keyword arguments to the worker
    )
    worker.run(audio_subset)
