FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv
RUN pip install gunicorn

WORKDIR /app
COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv lock
RUN pipenv install --system --deploy

COPY ["question_4_predict.py", "dv.bin", "model1.bin", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "question_4_predict:app"]