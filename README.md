\documentclass{article}

\usepackage{PRIMEarxiv}

\usepackage[utf8]{inputenc} % allow utf-8 input
\usepackage[T1]{fontenc}    % use 8-bit T1 fonts
\usepackage{hyperref}       % hyperlinks
\usepackage{url}            % simple URL typesetting
\usepackage{booktabs}       % professional-quality tables
\usepackage{amsfonts}       % blackboard math symbols
\usepackage{nicefrac}       % compact symbols for 1/2, etc.
\usepackage{microtype}      % microtypography
\usepackage{fancyhdr}       % header
\usepackage{graphicx}       % graphics
\graphicspath{{media/}}     % organize your images and other figures under media/ folder

%Header
\pagestyle{fancy}
\fancyhead[LO]{Few Shot Language Agnostic Keyword Spotting}
\rhead{} 
\thispagestyle{empty}

% Title
\title{Few Shot Language Agnostic Keyword Spotting (FSLAKWS)
\thanks{\textit{\underline{Citation}}: 
\textbf{Authors. Title. Pages.... DOI:000000/11111.}} 
}

\author{
  Ansh Singh, Arnav Raj, Suhani Soni, Aditya, Abhineet  \\
  Team Member \\
  Indian Institute of Technology Delhi \\
  New Delhi, India\\
  \texttt{\{ce1231156, cs5221652, mt6221981, ee3221760, mt1221736\}@iitd.ac.in} \\
   \And
  Anmol Goel \\
  Team Leader \\
  Indian Institute of Technology Delhi \\
  New Delhi, India\\
  \texttt{bb1221000@iitd.ac.in} \\
}

\begin{document}
\maketitle

\begin{abstract}
This document provides detailed information on a Few Shot Language Agnostic Keyword Spotting (FSLAKWS) framework using advanced audio processing and machine learning techniques. It covers scripts for timestamp extraction, inference pipelines, a Streamlit-based end-to-end UI, advanced audio embedding and Variational Autoencoder (VAE) workflows, and an audio augmentation pipeline.
\end{abstract}

% keywords can be removed
\keywords{Few Shot Learning \and Keyword Spotting \and Language Agnostic \and Audio Processing \and Machine Learning}

%-------------------
\section{Introduction}
This project implements a framework for Few Shot Language Agnostic Keyword Spotting (FSLAKWS) using advanced audio processing and machine learning techniques. The following sections provide documentation for each major script and workflow in the codebase.

%-------------------
\section{FSLAKWS: Scripts and Documentation}

\subsection{\texttt{extract\_timeStamps.py}}
\textbf{Overview:}  
This script provides utility functions for audio processing, keyword timestamp extraction, and accuracy evaluation.

\subsubsection{Functions}
\paragraph{\texttt{> cut\_audio\_segment :}}
Cuts a segment of audio from the specified start to end times and saves it to a given folder.  
\textbf{Parameters:}
\begin{itemize}
    \item \texttt{audio\_path (str)}: Path to the input audio file.
    \item \texttt{start\_time (float)}: Start time in seconds.
    \item \texttt{end\_time (float)}: End time in seconds.
    \item \texttt{output\_folder (str)}: Destination folder for the output segment.
    \item \texttt{segment\_name (str)}: Name of the saved segment.
\end{itemize}
\textbf{Returns:}  
\texttt{str}: Path to the saved audio segment.

\paragraph{\texttt{> group\_keywords :}}
Groups timestamps for each keyword from the input dictionary. \\ 
\textbf{Parameters:}
\begin{itemize}
    \item \texttt{input\_dict (dict)}: Dictionary containing audio metadata and timestamps.
\end{itemize}
\textbf{Returns:}  
\texttt{dict}: Grouped keywords with timestamps.  
\texttt{str}: File key associated with the input data.

\paragraph{\texttt{> extract\_keyword\_timestamps :}}
Identifies timestamped segments of keywords in an audio file using STFT-based processing.\\  
\textbf{Parameters:}
\begin{itemize}
    \item \texttt{audio\_path (str)}: Path to the audio file.
\end{itemize}
\textbf{Returns:}  
\texttt{list[tuple]}: A list of \texttt{(start\_time, end\_time)} tuples for detected segments.

\paragraph{\texttt{> accuracy :}}
Evaluates model predictions against ground truth.  \\
\textbf{Parameters:}
\begin{itemize}
    \item \texttt{GT\_actual (list)}: Ground truth data.
    \item \texttt{GT\_pred (list)}: Predicted keyword data.
\end{itemize}
\textbf{Returns:}  
\texttt{float}: Accuracy percentage.  
\texttt{list[float]}: Timestamp differences for errors.

%-------------------
\subsection{\texttt{inference.py}}
\textbf{Overview:}  
Defines classes and methods for preprocessing audio, extracting embeddings, and making keyword predictions.

\subsubsection{Classes}
\paragraph{\texttt{AudioPreprocessor :}}
Handles preprocessing of audio files, including resampling, normalization, and silence trimming.

\textbf{Parameters:}
\begin{itemize}
    \item \texttt{target\_sr (int)}: Target sampling rate (default: 16,000 Hz).
    \item \texttt{normalize (bool)}: Whether to normalize the audio (default: True).
    \item \texttt{trim\_silence (bool)}: Whether to trim silence (default: True).
    \item \texttt{max\_duration (Optional[float])}: Maximum duration of the audio in seconds.
    \item \texttt{mono (bool)}: Whether to convert audio to mono (default: True).
\end{itemize}

\textbf{Methods:}
\begin{itemize}
    \item \texttt{process}: Preprocesses the input audio file or array.
    \item \texttt{save\_processed\_audio}: Saves processed audio to a specified file.
\end{itemize}

\paragraph{\texttt{EmbeddingClassifier :}}
Defines a neural network for keyword classification based on audio embeddings.

\textbf{Parameters:}
\begin{itemize}
    \item \texttt{input\_size (int)}: Input size for the embedding vector.
    \item \texttt{num\_classes (int)}: Number of output classes.
\end{itemize}

\subsubsection{Global Objects}
\begin{itemize}
    \item \texttt{WhisperProcessor}: Processor for converting audio to features.
    \item \texttt{WhisperModel}: Encoder model from OpenAI’s Whisper.
    \item \texttt{EmbeddingClassifier}: Neural network classifier for keywords.
\end{itemize}

\subsubsection{Function \texttt{inference}}
Processes audio, extracts embeddings, and predicts the keyword.  
\textbf{Parameters:}
\begin{itemize}
    \item \texttt{model (nn.Module)}: Trained model for classification.
    \item \texttt{audio\_path (str)}: Path to the input audio file.
    \item \texttt{device (str)}: Device for computation (e.g., 'cpu' or 'cuda').
\end{itemize}
\textbf{Returns:}  
\texttt{int}: Predicted class ID.

%-------------------
\subsection{\texttt{main.py}}
\textbf{Overview:}  
Defines a Streamlit-based user interface for end-to-end processing of audio files and visualization of keyword detection.

\subsubsection{Workflow}
\begin{itemize}
    \item \textbf{Step 1: Upload a ZIP File}  
          Users upload a ZIP file containing .wav audio files. Files are extracted, and audio processing begins.
    \item \textbf{Step 2: Process Audio Files}  
          Extracts keyword timestamps using \texttt{extract\_keyword\_timestamps}. Audio segments are classified using the \texttt{EmbeddingClassifier}. Outputs are stored in a results list.
    \item \textbf{Step 3: Evaluate Accuracy}  
          Compares predicted results against a ground truth file (\texttt{dummy.json}). Computes overall accuracy and timestamp differences.
    \item \textbf{Step 4: Save Results}  
          Saves results in JSON and pickle formats for further analysis.
    \item \textbf{Step 5: Visualize Keyword Occurrences}  
          Displays a timeline of keyword occurrences. Provides a visual breakdown of keywords over time using \texttt{matplotlib}.
\end{itemize}

\subsubsection{Key Functions}
\paragraph{\texttt{process\_audio\_files :}}
Processes a list of audio files and predicts keywords.

\textbf{Parameters:}
\begin{itemize}
    \item \texttt{found\_wav\_files (list)}: List of .wav files to process.
    \item \texttt{id\_key (dict)}: Mapping of class IDs to keywords.
\end{itemize}

\textbf{Returns:}  
\texttt{list[dict]}: Results containing timestamps and predicted keywords.

\subsubsection{Dependencies}
\begin{itemize}
    \item Python libraries: \texttt{numpy, librosa, torch, soundfile, streamlit, matplotlib, zipfile, pickle}
    \item Pretrained model: \texttt{openai/whisper-base} (via Hugging Face Transformers)
\end{itemize}

\subsubsection{Usage}
\begin{verbatim}
pip install -r requirements.txt
streamlit run main.py
\end{verbatim}
Upload a ZIP file containing \texttt{.wav} files and view results in the web interface.

\subsubsection{Output}
\begin{itemize}
    \item JSON and pickle files containing processed results.
    \item Visualizations of keyword occurrences.
\end{itemize}

%-------------------
\section{Documentation for Advanced Audio Embedding and VAE Workflow}
This code integrates multiple components to process audio data, extract embeddings, and train a Variational Autoencoder (VAE). Below is a detailed explanation of the modules and their functionality.

\subsection{Loading and Preprocessing Dataset}
\textbf{Dataset Loading:}
\begin{verbatim}
ds = load_dataset("MLCommons/ml_spoken_words", 
                  languages=["ar"], 
                  split="train", 
                  trust_remote_code=True)
\end{verbatim}
\textbf{Parameters:}
\begin{itemize}
    \item \texttt{languages}: Specifies the language (e.g., Arabic: "ar").
    \item \texttt{split}: Chooses the dataset split (e.g., "train").
\end{itemize}

\subsection{Whisper Embedding Extraction}
\paragraph{\texttt{WhisperEmbeddingExtractor}}
Handles audio embedding extraction using OpenAI's Whisper model.

\textbf{Attributes:}
\begin{itemize}
    \item \texttt{model\_size (str)}: Size of the Whisper model (e.g., "small").
    \item \texttt{device (str)}: Computation device (‘cpu’ or ‘cuda’).
\end{itemize}

\textbf{Methods:}
\begin{itemize}
    \item \texttt{\_\_init\_\_}: Initializes the model and sets up the device.
    \item \texttt{extract\_embeddings}: Extracts embeddings from an audio signal.
\end{itemize}

\textbf{Function: \texttt{batch\_extract\_embeddings}}
Processes a dataset to extract embeddings in batches.

\subsection{Variational Autoencoder (VAE)}
\paragraph{\texttt{VAE}}
Defines the VAE architecture with convolutional layers for encoding and decoding.

\textbf{Attributes:}
\begin{itemize}
    \item \texttt{input\_channels (int)}
    \item \texttt{input\_height, input\_width (int)}
    \item \texttt{latent\_dim (int)}
    \item \texttt{hidden\_dims (list)}
\end{itemize}

\textbf{Methods:}
\begin{itemize}
    \item \texttt{encode}: Encodes input to latent space parameters (mean and log variance).
    \item \texttt{reparameterize}: Samples from the latent space using the reparameterization trick.
    \item \texttt{decode}: Reconstructs data from latent space.
    \item \texttt{forward}: Performs the full forward pass.
\end{itemize}

\subsubsection{Training and Validation}
\paragraph{\texttt{vae\_loss}}
Combines SSIM loss and KL divergence:
\[
    \mathrm{loss} = \mathrm{SSIM\_loss}(\hat{x}, x) + \alpha \cdot \mathrm{KL\_divergence}
\]

\paragraph{\texttt{train\_val\_vae}}
Trains the VAE with early stopping based on validation loss.

\subsubsection{Generating Samples}
\paragraph{\texttt{generate\_samples}}
Generates samples from the VAE's latent space.

\subsubsection{Utilities}
\begin{itemize}
    \item \texttt{save\_model}, \texttt{load\_model}
\end{itemize}

\subsubsection{Full Workflow Execution}
\begin{enumerate}
    \item Load the dataset.
    \item Extract embeddings using \texttt{WhisperEmbeddingExtractor}.
    \item Prepare the dataset for training.
    \item Train and validate the VAE.
    \item Save the trained model.
    \item Generate new samples from the latent space.
\end{enumerate}

\textbf{Output:}
\begin{itemize}
    \item Trained \texttt{VAE} model file (\texttt{vae\_model.pth}).
    \item Latent space embeddings.
    \item Generated samples for analysis.
\end{itemize}

%-------------------
\section{Documentation for Audio Augmentation and Dataset Generation Workflow}
This pipeline provides audio data augmentation to create a balanced dataset for machine learning tasks.

\subsection{Logging Configuration}
Log messages (INFO-level) are saved to \texttt{audio\_augmentation.log}.

\subsection{Parameters}
\begin{itemize}
    \item \texttt{TARGET\_COUNT = 1500}
    \item \texttt{AUGMENTATIONS\_PER\_FILE = 5}
    \item \texttt{MIN\_AUGMENTATIONS\_PER\_KEYWORD}, \texttt{MAX\_AUGMENTATIONS\_PER\_KEYWORD}
    \item \texttt{ALL\_AUGMENTATION\_FUNCTIONS}
\end{itemize}

\subsection{Augmentation Functions}
\paragraph{\texttt{add\_noise}} Adds Gaussian noise. 
\begin{itemize}
    \item \textbf{Parameters :}
    \begin{itemize}
        \item \texttt{noise\_factor} (float): Scaling factor for the noise (default: 0.005).
    \end{itemize}
\end{itemize}
\paragraph{\texttt{change\_volume}} Adjusts audio gain. 
\begin{itemize}
    \item \textbf{Parameters :}
    \begin{itemize}
        \item \texttt{gain} (float): Gain adjustment factor (default: 0.1).
    \end{itemize}
\end{itemize}
\paragraph{\texttt{add\_reverb}} Applies a reverb effect.
\begin{itemize}
    \item \textbf{Parameters :}
    \begin{itemize}
        \item \texttt{reverberance} (float): Strength of the reverb (default: 0.3).
    \end{itemize}
\end{itemize}
\paragraph{\texttt{compress\_audio}} Performs dynamic range compression.  
\begin{itemize}
    \item \textbf{Parameters :}
    \begin{itemize}
        \item \texttt{threshold} (float): Amplitude threshold for compression (default: 0.5).
        \item \texttt{ratio} (int) : Compression ratio (default: 4).
    \end{itemize}
\end{itemize}
\paragraph{\texttt{equalize}} Boosts low/high frequencies. 
\begin{itemize}
    \item \textbf{Parameters :}
    \begin{itemize}
        \item \texttt{low\_freq} (int): Lower frequency boost range.
        \item \texttt{high\_freq} (int):  Higher frequency boost range.
    \end{itemize}
\end{itemize}
\paragraph{\texttt{crop\_audio}} Crops a random audio segment.
\begin{itemize}
    \item \textbf{Parameters :}
    \begin{itemize}
        \item \texttt{crop\_duration } (float): Duration of the cropped segment in seconds (default: 0.1).
    \end{itemize}
\end{itemize}
\paragraph{\texttt{shift\_audio}} Shifts the audio in time. 
\begin{itemize}
    \item \textbf{Parameters :}
    \begin{itemize}
        \item \texttt{shift\_max} (float): Duration of the cropped segment in seconds (default: 0.1).
    \end{itemize}
\end{itemize}
\paragraph{\texttt{time\_mask}} Applies a time-based mask by zeroing out a segment. 
\begin{itemize}
    \item \textbf{Parameters :}
    \begin{itemize}
        \item \texttt{mask\_size} (float): Duration of the mask in seconds (default: 0.1).
    \end{itemize}
\end{itemize}
\paragraph{\texttt{frequency\_mask}} Zeroes out a range of frequencies. 
\begin{itemize}
    \item \textbf{Parameters :}
    \begin{itemize}
        \item \texttt{mask\_size} (float): Fraction of frequencies to mask (default: 0.1).
    \end{itemize}
\end{itemize}
\paragraph{\texttt{dynamic\_range\_compression}} Another compression approach with threshold/ratio.
\begin{itemize}
    \item \textbf{Parameters :}
    \begin{itemize}
        \item \texttt{threshold} (float): Amplitude threshold.
        \item \texttt{ratio} (float): Compression ratio.
    \end{itemize}
\end{itemize}

\subsection{Augmentation Workflow}
\paragraph{\texttt{augment\_audio}}
Randomly selects and applies augmentation functions to the input audio.
\begin{itemize}
    \item \textbf{Parameters :}
    \begin{itemize}
        \item \texttt{audio} : Input audio signal.
        \item \texttt{sr} : Sampling rate of the audio.
        \item \texttt{selected\_augmentations} : List of augmentation functions to choose from.
    \end{itemize}
    \item \textbf{Returns :}
    \begin{itemize}
        \item Augmented audio signal.
    \end{itemize}
\end{itemize}

\subsection{Keyword Folder Processing}
\paragraph{\texttt{process\_keyword\_folder}}
Ensures the target file count is met by generating additional augmented files if needed.
\begin{itemize}
    \item \textbf{Parameters :}
    \begin{itemize}
        \item \texttt{keyword\_input\_path} : Path to the input folder.
        \item \texttt{keyword\_output\_path} : Path to the output folder.
        \item \texttt{selected\_augmentations} : List of augmentations to apply.
    \end{itemize}
    \item \textbf{Workflow :}
    \begin{itemize}
        \item Normalize folder paths to handle non-ASCII characters.
        \item Copy original files to the output folder.
        \item Calculate the number of augmented files required.
        \item Generate augmented files using the specified augmentations.
    \end{itemize}
\end{itemize}


\subsection{Main Execution}
\paragraph{\texttt{main}}
Iterates over keyword folders, applies augmentations, and writes augmented files to the output directory.
\begin{itemize}
    \item \textbf{Parameters :}
    \begin{itemize}
        \item \texttt{input\_dataset\_path} : Path to the dataset containing keyword folders.
        \item \texttt{output\_dataset\_path} :  Path to the output dataset directory.
    \end{itemize}
    \item \textbf{Workflow :}
    \begin{itemize}
        \item Normalize input and output dataset paths.
        \item Identify keyword folders in the input dataset.
        \item Randomly select augmentations for each keyword.
        \item Call \texttt{process\_keyword\_folder} for each keyword folder.
    
    \end{itemize}
    
\end{itemize}
\subsection{Example Execution}
\subsubsection{Setup}
Replace the placeholder paths in the \texttt{if \_\_name\_\_ == "\_\_main\_\_"}: block with your dataset paths:\\
input\_dataset\_path = "/path/to/input\_dataset"\\
output\_dataset\_path = "/path/to/output\_dataset"
\subsubsection{Run the Script}
\begin{verbatim}
python audio_augmentation.py
\end{verbatim}
\textbf{Output:} Augmented files in the specified output dataset directory and logs in \texttt{audio\_augmentation.log}.

%-------------------
\section{Conclusion}
By combining the scripts described above, this project provides a comprehensive framework for Few Shot Language Agnostic Keyword Spotting and advanced audio workflows. From audio timestamp extraction and inference to augmentations and VAE-based embedding generation, the pipeline can serve as a robust foundation for scalable audio processing and analysis tasks.


\end{document}
