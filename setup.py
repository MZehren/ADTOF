import setuptools
with open("README.rst", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='ADTOF',  
     version='0.1',
     scripts=['ADTOF'] ,
     author="Mickael zehren",
     author_email="mickael.zehren@gmail.com",
     description="Automatic drums transcription database conversion",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/MZehren/ADTOF",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
     ],
 )