import os
from typing import List, Tuple, Optional
import math
from transformers import AutoModel, AutoTokenizer, AutoConfig
import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from logger import get_logger

class ProteinEmbedder:
    def __init__(
            self,
            device: str = None,
            model_name: str = None,
            hf_path: str = None,
            dataset: pd.DataFrame = None,
            column_seq: str = None,
            column_label: str = None,
            columns_ignore: List[str] = None
            ):
        """
        Initializes the protein embedding model.
        
        :param device: Device to load the model on ("cuda" or "cpu"). If None, auto-detect.
        :param hf_path: Hugging Face model identifier (e.g., "facebook/esm2_t6_8M_UR50D")
        :param dataset: Pandas DataFrame with the dataset to embed.
        :param column_seq: Name of the column with the sequences.
        :param column_label: Name of the column with the labels (optional).
        :param columns_ignore: List of columns to not include in the output DataFrame.
        """
   
        self.model_name = model_name
        self.hf_path = hf_path
        self.dataset = dataset
        self.column_seq = column_seq
        self.column_label = column_label
        self.columns_ignore = columns_ignore
        
        self.tokenizer = None
        self.device = None
        self.model = None
        self.embeddings = None

        self.logger = get_logger(f"ProteinEmbedder_{hf_path if hf_path else model_name}")

        self.__selectDevice(device)

    def __selectDevice(self, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
    
    def loadModelTokenizer(self):
        """
        Loads the model and tokenizer from Hugging Face.
        :raises ValueError: If the model name is not provided.
        :raises Exception: If the model or tokenizer cannot be loaded.
        """
        if self.hf_path is None:
            msg = "Model Name is not provided. Please provide a valid Hugging Face model name."
            if self.model_name is not None:
                msg += f" A model name was provided: {self.model_name}. We suggest to use loadModel() method instead."
            self.logger.error(msg)
            raise ValueError(msg)

        config = AutoConfig.from_pretrained(self.hf_path)
        self.logger.info(f"Loading model and tokenizer for {self.hf_path}")
        self.logger.info(f"Model architecture: {config.architectures[0]}")
        if "T5" in config.architectures[0]:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.hf_path, 
                    trust_remote_code=True,
                    do_lower_case=False
                )
            except TypeError as e:
                # TypeError: Descriptors cannot be created directly.
                if "Descriptors cannot be created directly." in str(e):
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.hf_path, 
                        trust_remote_code=True,
                        use_fast=False
                    )
                else:
                    raise e
        
            self.model = AutoModel.from_pretrained(self.hf_path, trust_remote_code=True, config=config).to(self.device)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_path, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.hf_path, trust_remote_code=True, config=config).to(self.device)
        
        self.model.eval()
        self.logger.info(f"Model and tokenizer for {self.hf_path} loaded successfully.")
    
    def loadModel(self):
        """
        Loads imported or local model (not implemented yet).
        """
        if self.model_name is None:
            msg = "Model Name is not provided. Please provide a valid model name."
            if self.hf_path is not None:
                msg += f" A model name was provided: {self.hf_path}. We suggest to use loadModelTokenizer() method instead."
            self.logger.error(msg)
            raise ValueError(msg)
        
        self.logger.info(f"Loading model {self.model_name}")

        if "esmc" in self.model_name:
            self.model = ESMC.from_pretrained(self.model_name).to(self.device)
        else:
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.logger.info(f"Model {self.model_name} loaded successfully.")


    def pad_sequence(self, sequence: str, target_length: int, pad_char: str = "X") -> str:
        """Pads or truncates a protein sequence to a fixed length."""
        return sequence[:target_length].ljust(target_length, pad_char)
        
    def showMemoryUsage(self):
        torch.cuda.synchronize()
        self.logger.info("Current memory usage: {:.2f} MB".format(torch.cuda.memory_allocated() / (1024 * 1024)))
    
    def cleanGPUMemory(self):
        """
        Cleans GPU memory by deleting model, tokenizer, and embeddings.
        This is useful to avoid memory leaks and free up resources.
        """
        self.logger.info("Cleaning GPU memory")
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if self.embeddings is not None:
            del self.embeddings
        torch.cuda.empty_cache()

    # TODO 1024? Por que 1024
    def embeddingTokenBatch(self, sequences: List[str], max_length: int = 1024) -> torch.Tensor:
        """
        Generates embeddings for a batch of sequences. Exclusive use of the tokenizer.
        :param sequences: List of sequences to embed.
        :param max_length: Maximum length of sequences after tokenization.
        """
        inputs = self.tokenizer(
            sequences, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            add_special_tokens=False,
            max_length=max_length
            ).to(self.device)
        
        # Check the architecture of the model and process accordingly
        with torch.no_grad():
            # Try the batch, if it fails by CUDA OOM, it means the batch size is too large
            try:
                if "Bert" in self.model.config.architectures[0]:
                    outputs = self.model(
                        input_ids=inputs["input_ids"],
                        output_hidden_states=False
                    )

                elif "Esm" in self.model.config.architectures[0]:
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    outputs = self.model(**inputs, output_hidden_states=False)

                elif "Mixtral" in self.model.config.architectures[0]:
                    outputs = self.model(
                            input_ids=inputs["input_ids"], 
                            output_hidden_states=True)
                    
                elif "T5" in self.model.config.architectures[0]:
                    outputs = self.model.encoder(
                        input_ids=inputs["input_ids"],
                        output_hidden_states=False
                    )
    
                else:
                    outputs = self.model(**inputs)

                # Clear memory
                if inputs is not None:
                    del inputs

                return outputs.last_hidden_state 
            
            except torch.cuda.OutOfMemoryError as e:
                self.logger.error("CUDA OOM detected. Consider reducing the batch size.")
                del inputs
                torch.cuda.empty_cache()
                raise
            except Exception as e:
                self.logger.error(f"Error processing embeddings: {e}")
                raise

    def embeddingBatch(self, sequences: List[str]) -> torch.Tensor:
        """
        Generates embeddings for a batch of sequences. Uses the model.
        :param sequences: List of sequences to embed.
        :param max_length: Maximum length of sequences after tokenization.
        """
        with torch.no_grad():
            
            # Initialize a empty torch tensor to store the embeddings
            outputs = []
            # Process each sequence in the batch
            for sequence in sequences:
                try:
                    if "esmc" in self.model_name:
                        protein = ESMProtein(sequence=sequence)
                        protein_tensor = self.model.encode(protein).to(self.device)

                        EMBEDDING_CONFIG = LogitsConfig(return_embeddings=True, return_hidden_states=True)
                        logits_output = self.model.logits(protein_tensor, EMBEDDING_CONFIG)

                        # Append the embeddings to the tensor
                        outputs.append(logits_output.embeddings)
                    else:
                        self.logger.error("Model not supported for embeddingBatch.")
                        return None
                except Exception as e:
                    self.logger.error(f"Error processing sequence {sequence}: {e}")
                    # Empty cache
                    torch.cuda.empty_cache()
                    raise

            if outputs:
                return torch.cat(outputs, dim=0)
            else:
                self.logger.error("No outputs generated. Check the input sequences.")
                return None
                

    def embeddingProcess(self, batch_size: int = 50, max_length: Optional[int] = None):
        """
        Generates embeddings for the entire dataset in batches.
        :param batch_size: Size of each batch for processing.
        """
        self.logger.info("Starting embedding process")
        self.logger.info(f"Batch size: {batch_size}")

        embeddings_list = []
        if max_length is None:
            target_length = max(len(seq) for seq in self.dataset[self.column_seq])
        else:
            target_length = max_length

        if self.hf_path is not None:
            sequences = self.dataset[self.column_seq].tolist()
        elif self.model_name is not None:
            sequences = [self.pad_sequence(seq, target_length) for seq in self.dataset[self.column_seq]]
        else:
            raise ValueError("No valid model or Hugging Face path nor model name provided.")

        # TODO Aca se hizo un cambio de self.dataset a padded_sequences. Comprobar si esta bien
        progress_bar = tqdm(
            range(0, len(sequences), batch_size), 
            desc="Embedding progress", 
            total=math.ceil(len(sequences) / batch_size),
            dynamic_ncols=True,
            ncols=0
        )
        for i in progress_bar:
            # Generate batch
            #batch = self.dataset[self.column_seq].iloc[i:i + batch_size]
            batch = sequences[i:i + batch_size]

            # Process batch via Hugging Face
            if self.hf_path is not None:
                last_hidden_layer = self.embeddingTokenBatch(batch)
            # Process batch via model
            else:
                last_hidden_layer = self.embeddingBatch(batch)
            
            # Moving to CPU and converting to numpy
            batch_embedding = last_hidden_layer.mean(dim=1).cpu().numpy()
            embeddings_list.append(batch_embedding)
            
            # Clear memory
            del batch_embedding
            torch.cuda.empty_cache()

        
        self.logger.info("Embedding process finished")

        self.cleanGPUMemory()

        # All batches previously processed are concatenated
        self.embeddings = np.concatenate(embeddings_list, axis=0)
        self.logger.info(f"Embeddings stored in memory: {self.embeddings.shape}")
    
    def showEmbeddings(self):
        """
        Displays the embeddings in the logger.
        :raises ValueError: If no embeddings are available.
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available. Please run the embedding process first.")
        
        self.logger.info(f"Embeddings: {self.embeddings}")

    def getDataFrame(self):
        """
        Converts the embeddings to a DataFrame.
        :raises ValueError: If no embeddings are available.
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available. Please run the embedding process first.")
        
        # Generate a Dataframe with the non ignored columns
        df_embedding = self.dataset.copy()
        df_embedding = df_embedding.drop(columns=self.columns_ignore, errors='ignore')

        df_embedding = df_embedding.join(
            pd.DataFrame(self.embeddings, columns=[f"p_{i}" for i in range(self.embeddings.shape[1])]),
        )

        #df_embedding = pd.DataFrame(self.embeddings)
        #df_embedding.columns = [f"p_{i}" for i in range(df_embedding.shape[1])]

        return df_embedding
    
    def reduceDimensionality(self, tensor, n_components: int = 2, random_state: int = 42, method: str = 'pca'):
        """
        Reduces the dimensionality of the tensor using PCA or t-SNE.
        :param tensor: Input tensor to reduce.
        :param n_components: Number of components to keep.
        :param random_state: Random state for reproducibility.
        :param method: Method to use for dimensionality reduction ('pca' or 'tsne').
        :return: Reduced tensor.
        """
        if method == 'pca':
            pca = PCA(n_components=n_components)
            return pca.fit_transform(tensor)
        elif method == 'tsne':
            tsne = TSNE(n_components=n_components, random_state=random_state)
            return tsne.fit_transform(tensor)
        else:
            raise ValueError(f"Invalid method: {method}")

    def plotDensityPCA(self,
                        resolution: Tuple[int, int] = (1200, 800),
                        dpi: int = 100, 
                        point_size: int = 10,
                        alpha_range: Tuple[float, float] = (0.1, 0.9),
                        bandwidth_scale: float = 0.1,
                        color_palette = 'husl',
                        title:str = None,
                        xlabel: str = 'PCA 1',
                        ylabel: str = 'PCA 2',
                        dim_reduction_method: str = 'pca',
                        legend_marker_size: int = 6):
        """
        Generates a scatter plot of embeddings after PCA reduction, with point density visualization.
        
        Parameters:
        - resolution: Tuple (width, height) in pixels for the desired image
        - dpi: Dots per inch for density calculation
        - label_col: Name of the column containing labels for coloring (optional)
        - point_size: Base size of points
        - alpha_range: Tuple with (min alpha, max alpha) for density
        - bandwidth_scale: Scaling factor for KDE bandwidth
        - color_palette: Seaborn color palette name or list of colors
        - title: Plot title (optional, won't show if None)
        - xlabel: Label for x-axis
        - ylabel: Label for y-axis
        - dim_reduction_method: Method for dimensionality reduction ('pca' or 'tsne')
        - legend_marker_size: Size of the markers in the legend
        """
        # Normalize
        #feature_cols = [col for col in embeddings_df.columns if col.startswith('p_')]
        #X = embeddings_df[feature_cols].values
        X = StandardScaler().fit_transform(self.embeddings)
        
        reduced = self.reduceDimensionality(X, n_components=2, random_state=42, method=dim_reduction_method)
        df_pca = pd.DataFrame(data=reduced, columns=['PCA_1', 'PCA_2'])
        
        # Add sequence and label columns
        df_pca['sequence'] = self.dataset[self.column_seq].values
        if self.column_label is not None and self.column_label in self.dataset.columns:
            df_pca['label'] = self.dataset[self.column_label].values
        
        width_pixels, height_pixels = resolution
        width_inches = width_pixels / dpi
        height_inches = height_pixels / dpi
        
        # Calculate axis range to determine discretization
        x_range = df_pca['PCA_1'].max() - df_pca['PCA_1'].min()
        y_range = df_pca['PCA_2'].max() - df_pca['PCA_2'].min()
        
        # Calculate the size of a "pixel" in data units
        x_pixel_size = x_range / (width_inches * dpi)
        y_pixel_size = y_range / (height_inches * dpi)
        
        # Discretize coordinates to group nearby points
        df_pca['PCA_1_discrete'] = np.round(df_pca['PCA_1'] / x_pixel_size) * x_pixel_size
        df_pca['PCA_2_discrete'] = np.round(df_pca['PCA_2'] / y_pixel_size) * y_pixel_size
        
        # Calculate the bandwidth based on the scale factor
        xy = np.vstack([df_pca['PCA_1'], df_pca['PCA_2']])
        print(xy.shape)
        try:
            kde = gaussian_kde(xy, bw_method=bandwidth_scale)
            density = kde(xy)
        except LinAlgError as e:
            self.logger.error("Not enough variance in the data for KDE.")
            raise e
        
        # Normalize density for alpha values
        density_norm = (density - density.min()) / (density.max() - density.min())
        df_pca['alpha'] = alpha_range[0] + (alpha_range[1] - alpha_range[0]) * density_norm
        
        # Create the plot
        plt.figure(figsize=(width_inches, height_inches), dpi=dpi)
        
        if self.column_label is not None and self.column_label in self.dataset.columns:
            # Plot with color by label
            unique_labels = df_pca['label'].unique()
            palette = sns.color_palette(color_palette, n_colors=len(unique_labels))
            
            legend_elements = []
            
            for i, label in enumerate(unique_labels):
                mask = df_pca['label'] == label
                subset = df_pca[mask]
                plt.scatter(subset['PCA_1'], subset['PCA_2'], 
                            s=point_size, 
                            alpha=subset['alpha'],
                            color=palette[i],
                            label='_nolegend_')
                
                legend_elements.append(plt.Line2D([0], [0], 
                                                marker='o', 
                                                color='w',
                                                label=str(label),
                                                markerfacecolor=palette[i],
                                                markersize=legend_marker_size))
            
            plt.legend(handles=legend_elements, title=self.column_label)
        else:
            # Plot without label coloring
            plt.scatter(df_pca['PCA_1'], df_pca['PCA_2'], 
                        s=point_size, 
                        color=sns.color_palette(color_palette)[0],
                        alpha=df_pca['alpha'])
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if title is not None:
            plt.title(title)

        plt.tight_layout()
        return plt.gcf(), df_pca
    
    def saveEmbeddings(self, path: str):
        """
        Saves the embeddings to a file.
        :param path: Path to save the embeddings.
        :raises ValueError: If no embeddings are available.
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available. Please run the embedding process first.")
        
        # Save embeddings (numpy) compressed
        np.savez_compressed(path, embeddings=self.embeddings)
        self.logger.info(f"Embeddings saved to {path}")

    def loadEmbeddings(self, path: str):
        """
        Loads the embeddings from a file.
        :param path: Path to the file containing the embeddings.
        :raises FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")
        
        # Load embeddings (numpy)
        data = np.load(path)
        self.embeddings = data['embeddings']
        self.logger.info(f"Embeddings loaded from {path}")
