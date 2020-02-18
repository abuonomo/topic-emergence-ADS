FROM python:3.7.6

WORKDIR home
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY static ./static
COPY templates/ ./templates
COPY app.py ./
RUN chown -R www-data:www-data .

# Label image with git commit url
ARG GIT_URL=unspecified
ARG VERSION=unspecified
LABEL org.label-schema.schema-version=1.0
LABEL org.label-schema.url=$GIT_URL
LABEL org.label-schema.version=$VERSION
ENV VERSION=$VERSION
ENV GIT_URL=$GIT_URL

RUN mkdir data
ENV APP_DATA_DIR=data
USER www-data
CMD gunicorn app:app -b :5000 --timeout 1200