docker build \
    -f Dockerfile.base \
    -t anthesevenants/stats:base .

docker build \
    -f Dockerfile.rstudio \
    -t anthesevenants/stats:rstudio .
