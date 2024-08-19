from setuptools import setup, find_packages

setup(
    name="tara",
    version="0.1.0",
    author="eliird",
    author_email="irdali1996@gmail.com",
    description="Text and Audio Emotion Detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/eliird/tara",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
    ],
    entry_points={
        'console_scripts': [
        ],
    },
    include_package_data=True,
    package_data={

    },
)
