from distutils.core import setup

setup(
    name='gtcacs',  # How you named your package folder (MyLib)
    packages=['gtcacs'],  # Chose the same as "name"
    version='0.0.2',
    license='MIT',
    description='Generative Text Compression with Agglomerative Clustering Summarization (GTCACS)',
    author='Andrea Lenzi',
    author_email='andrealenzi11@gmail.com',
    url='https://github.com/andrealenzi11/gen-text-compr-aggl-clust-sum.git',
    download_url='https://github.com/andrealenzi11/gen-text-compr-aggl-clust-sum/archive/0.0.1.tar.gz',
    keywords=[
        'topic',
        'topic modeling',
        'topic modelling',
        'topic extraction',
        'topics',
        'topics modeling',
        'topics modelling',
        'topics extraction',
    ],
    install_requires=[
        'tensorflow',
        'scikit-learn',
        'numpy',
        'scipy',
        'tqdm',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3',  # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
