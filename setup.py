from setuptools import setup

setup(name='MARM',
      version='2.0',
      description='Melanoma Adaptive Resistance Model',
      author='',
      packages=['MARM'],
      install_requires=[
            'numpy',
            'petab',
            'amici==0.11.18',
            'pypesto==0.2.7',
            'fides==0.5.1',
            'astropy',
            'mizani',
            'plotnine',
            'pandas',
            'matplotlib',
            'seaborn',
            'xlrd',
            'snakemake',
            'tabulate',
            'sklearn',
            'pysb@https://github.com/FFroehlich/pysb@energy_modeling',
            'scipy',
            'networkx'
      ],
      python_requires='>=3.7',
      zip_safe=False)
