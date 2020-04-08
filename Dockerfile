FROM python:3.7 as base

# Install dependencies
COPY requirements.txt home/
RUN pip install -r home/requirements.txt &&\
    pip install tslearn==0.3.0 &&\
    python -m spacy download en_core_web_sm

# Add src
COPY Makefile recipes.py home/
COPY src home/src

# Label image with git commit url
ARG GIT_URL=unspecified
ARG VERSION=unspecified
LABEL org.label-schema.schema-version=1.0
LABEL org.label-schema.url=$GIT_URL
LABEL org.label-schema.version=$VERSION
ENV VERSION=$VERSION

# Run as appuser
#RUN groupadd -g 999 user && \
#    useradd -r -u 999 -g user user
#RUN chown -R user:user home/
#USER user
WORKDIR /home/

ENTRYPOINT ["make"]