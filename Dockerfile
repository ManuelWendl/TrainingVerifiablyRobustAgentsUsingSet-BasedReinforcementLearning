# get image (default CORA image)
FROM tobiasladnertum/cora:r2024b
WORKDIR /

# ./code
COPY code /code
RUN sudo chown -R matlab /code

# ./results
COPY results /results
RUN sudo chown -R matlab /results

# Licensing
# Option 1: Licence server: Nothing to do here.
# Option 2: License file: Copy file into docker
COPY license.lic /home/matlab/Documents/MATLAB

# set up user 'matlab'
USER matlab
WORKDIR /
