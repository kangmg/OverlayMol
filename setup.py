from setuptools import setup, find_namespace_packages

setup(
    name='OverlayMol',
    version='0.0.1',
    author='Kang mingi',
    author_email='kangmg@korea.ac.kr',
    description='Overlay and align molecular structures for comparison and analysis.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  
    url='https://github.com/kangmg/OverlayMol',
    keywords=['chemistry','computational chemistry', 'visualizer', "molecule", "graph"],
    #include_package_data=True,
    packages=find_namespace_packages(), 
    install_requires=[
        "matplotlib",
        "plotly",
        "numpy"
    ],
    classifiers=[ 
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Chemistry'
    ],
)
