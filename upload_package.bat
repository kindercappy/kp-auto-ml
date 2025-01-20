@echo on

REM Navigate to the scripts directory and run your Python script
@REM cd scripts
python packaging.py
@REM cd ..

@REM REM Navigate to the package directory and build the package
cd packaging
python -m build
@REM cd ..

@REM REM Upload the package to a test Python environment
@REM REM Replace 'YOUR_REPOSITORY_URL' with your repository URL
@REM REM Replace 'YOUR_API_TOKEN' with your API token
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/* -u __token__ -p %TEST_PYPI_API_TOKEN% --verbose
@REM python -m twine upload dist/* -u __token__ -p %PYPI_API_TOKEN% --verbose

@REM REM Cleanup build files (optional)
rd /s /q packaging

pause