# Guidelines

- Check the [Gitter](https://gitter.im/KordingLab/spykes) to discuss the issue

# Testing

Install the testing dependencies:

```bash
pip install flake8 nose
```

Make sure that tests are passing for both Python 2.7 and Python 3.6:

```bash
nosetests tests/ --with-coverage  # Unit tests
flake8 --count spkyes  # Linting
```

Making sure these steps pass will help the continuous integration step go smoothly.
