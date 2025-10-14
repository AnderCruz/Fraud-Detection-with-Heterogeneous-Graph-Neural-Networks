# Fraud Detection with Heterogeneous Graph Neural Networks

This repository contains a Jupyter Notebook implementation of a fraud detection model using Heterogeneous Graph Neural Networks (HAN) with PyTorch and PyTorch Geometric. The model processes synthetic transaction data to identify fraudulent transactions based on user, account, device, and transaction relationships.

## Overview

The project simulates a heterogeneous graph with four node types (users, accounts, devices, and transactions) and their relationships. It uses a HAN model to learn embeddings for transactions and classify them as fraudulent or non-fraudulent. The embeddings are visualized using t-SNE to highlight the separation between fraud and normal transactions.

Key components:
- **Data Generation**: Synthetic data with realistic patterns for users, accounts, devices, and transactions.
- **Graph Structure**: Heterogeneous graph with edges connecting users to accounts, accounts to transactions, users to devices, and devices to transactions.
- **Model**: A HAN-based neural network for fraud detection.
- **Training**: Training loop with Adam optimizer and CrossEntropyLoss.
- **Evaluation**: Accuracy calculation for fraud detection.
- **Visualization**: t-SNE plot of transaction embeddings.

## Requirements

To run the notebook, ensure you have the following dependencies installed:

- Python 3.8+
- PyTorch (compatible version with PyTorch Geometric)
- PyTorch Geometric
- NumPy
- scikit-learn
- Matplotlib

Install the dependencies using pip:

```bash
pip install torch torch-geometric numpy scikit-learn matplotlib
```

**Note**: Check the [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/en/latest/) for compatible PyTorch versions.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/fraud-detection-han.git
   cd fraud-detection-han
   ```

2. Set up a virtual environment (optional but recommended):

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Alternatively, install the packages listed above manually.

4. Ensure you have Jupyter Notebook or JupyterLab installed:

   ```bash
   pip install jupyter
   ```

## Usage

1. Open the Jupyter Notebook:

   ```bash
   jupyter notebook fraud_detection.ipynb
   ```

2. Run the cells in sequence:
   - **Cell 1**: Import required libraries.
   - **Cell 2**: Generate synthetic data for users, accounts, devices, and transactions.
   - **Cell 3**: Define the HAN model.
   - **Cell 4**: Train the model for 150 epochs.
   - **Cell 5**: Evaluate the model and print accuracy.
   - **Cell 6**: Visualize transaction embeddings using t-SNE.

3. View the output:
   - Training progress is printed every 30 epochs.
   - Final accuracy is displayed after evaluation.
   - A t-SNE plot shows transaction embeddings, with fraudulent transactions in red and normal transactions in blue.

## File Structure

- `fraud_detection.ipynb`: Jupyter Notebook containing the complete implementation.
- `README.md`: This file, providing project documentation.
- `requirements.txt`: List of required Python packages (optional, create manually if needed).

## Notes

- The synthetic data is randomly generated, so results may vary between runs. Set a random seed (e.g., `np.random.seed(42)` and `torch.manual_seed(42)`) for reproducibility.
- Ensure compatibility between PyTorch and PyTorch Geometric versions to avoid runtime errors.
- The t-SNE visualization requires `%matplotlib inline` for inline plotting in Jupyter Notebook. If running in a different environment, adjust the plotting backend as needed.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.


## Contact

For questions or feedback, please open an issue on GitHub or contact [your-email@example.com].
```

### Instructions for Use
1. **Copy the Content**:
   - Copy the entire Markdown content above.
   
2. **Paste into GitHub**:
   - **Web Interface**: Go to your GitHub repository, click on `README.md` (or create a new file named `README.md` if it doesn’t exist), click the pencil icon to edit, paste the content, and commit the changes.
   - **Local Repository**: Save the content as `README.md` in your repository’s root directory, then commit and push:
     ```bash
     git add README.md
     git commit -m "Add README for fraud detection project"
     git push origin main
     ```

3. **Update Placeholder**:
   - Replace `your-username` in the clone URL with your actual GitHub username.
   - Replace `[your-email@example.com]` with your contact email or remove the contact section if not needed.

4. **Optional: Add `requirements.txt`**:
   - Create a `requirements.txt` file with the following content to match the README instructions:
     ```
     torch
     torch-geometric
     numpy
     scikit-learn
     matplotlib
     jupyter
     ```
   - Add it to your repository:
     ```bash
     git add requirements.txt
     git commit -m "Add requirements.txt"
     git push origin main
     ```

5. **Verify**:
   - Check your GitHub repository’s main page to ensure the README renders correctly.
   - Test the notebook locally to confirm all instructions work as expected.
