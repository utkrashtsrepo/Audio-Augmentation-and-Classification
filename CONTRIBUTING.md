# Contributing to Audio Data Augmentation

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear title and description
- Steps to reproduce the bug
- Expected vs actual behavior
- Your environment (OS, Python version, library versions)
- Any relevant error messages or logs

### Suggesting Enhancements

We welcome suggestions for new features or improvements:
- Open an issue describing the enhancement
- Explain why it would be useful
- Provide examples if possible

### Pull Requests

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/audio-data-augmentation.git
   cd audio-data-augmentation
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation if needed
   - Test your changes

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: Description of your changes"
   ```
   
   Use clear commit messages:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Update:` for updates to existing features
   - `Docs:` for documentation changes
   - `Refactor:` for code refactoring

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   - Provide a clear title and description
   - Reference any related issues
   - Wait for review and feedback

## Code Style Guidelines

### Python Code
- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused on a single task
- Maximum line length: 100 characters

### Example:
```python
def add_white_noise(signal, noise_factor=0.02):
    """
    Add white noise to audio signal.
    
    Args:
        signal (np.ndarray): Input audio signal
        noise_factor (float): Noise intensity factor
        
    Returns:
        np.ndarray: Augmented audio signal
    """
    noise = np.random.normal(0, signal.std(), signal.size)
    return signal + noise * noise_factor
```

### Notebooks
- Use clear markdown cells to explain code
- Organize cells logically
- Remove unnecessary output before committing
- Add comments for complex operations

## Testing

Before submitting a PR:
- Test your code with different audio files
- Ensure no errors occur
- Verify outputs are as expected
- Check that existing functionality still works

## Documentation

- Update README.md if adding new features
- Add docstrings to new functions
- Update examples if API changes
- Keep documentation clear and concise

## Questions?

Feel free to open an issue for any questions or clarifications.

Thank you for contributing! 🎉

