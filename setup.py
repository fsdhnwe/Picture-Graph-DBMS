from setuptools import setup, find_packages

setup(
    name="img_neo4j",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "pytube>=15.0.0",
        "neo4j>=5.0.0",
        "numpy>=1.24.0",
        "python-dotenv>=1.0.0",
        "sentence-transformers>=2.2.0",
        "llama-cpp-python>=0.2.0",
    ],
    author="what ever you want",
    author_email="your.email@example.com",
    description="A multimodal RAG system using Neo4j for image and video processing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/fsdhnwe/Picture-Graph-RAG",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)