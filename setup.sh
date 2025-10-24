```bash
#!/bin/bash
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "Dependencies installed successfully."
else
    echo "Error installing dependencies. Check requirements.txt for issues."
    exit 1
fi
```
