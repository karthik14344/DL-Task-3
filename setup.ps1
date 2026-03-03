# Setup script for Lab Task 3
# Run: .\setup.ps1

Write-Host "Creating virtual environment..." -ForegroundColor Green
python -m venv venv

Write-Host "Activating virtual environment..." -ForegroundColor Green
.\venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..." -ForegroundColor Green
pip install --upgrade pip

Write-Host "Installing PyTorch with CUDA 12.8..." -ForegroundColor Green
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Write-Host "Installing remaining dependencies..." -ForegroundColor Green
pip install transformers accelerate datasets bitsandbytes peft sentencepiece
pip install rouge-score nltk evaluate
pip install langchain langchain-community langchain-huggingface faiss-cpu sentence-transformers pypdf unstructured
pip install matplotlib pandas numpy tqdm

Write-Host "Downloading NLTK data..." -ForegroundColor Green
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Activate the env with: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
