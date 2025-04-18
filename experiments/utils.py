import pickle
import os

def save_processed_data(filename, *arrays):
    """Save processed data to a pickle file."""
    # Check if in Colab
    in_colab = False
    try:
        import google.colab
        in_colab = True
    except ImportError:
        pass

    # Set appropriate path based on environment
    if in_colab:
        print(
            "Enter path to the 281_final_project_data folder in Drive (e.g., /content/drive/MyDrive/281_final_project_data):")
        path = input()
    else:
        path = "./281_final_project_data"

    # Make sure directory exists
    os.makedirs(path, exist_ok=True)

    # Save to appropriate path
    file_path = os.path.join(path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(arrays, f)

    print(f"Data saved to {file_path}")
    # Note: Requires manual upload to Google Drive after saving locally and you are not using Colab


def load_processed_data(filename):
    """
    Load processed data from a pickle file.

    Parameters:
    filename: filename of the pickle file

    Returns:
    tuple: All arrays that were saved
    """
    # Check if in Colab
    in_colab = False
    try:
        import google.colab
        in_colab = True
    except ImportError:
        pass

    # Set appropriate path based on environment
    if in_colab:
        path = "/content/drive/281_final_project_data"
    else:
        path = "./281_final_project_data"

    file_path = os.path.join(path, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load the data
    with open(file_path, 'rb') as f:
        arrays = pickle.load(f)

    return arrays