FROM registry.cn-shanghai.aliyuncs.com/tcc_public/python:3.10

RUN /usr/local/bin/python -m pip install --upgrade pip && \
    pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install regex==2021.11.10 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install catboost -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install scipy -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    apt -y update &&  \
    apt install zip


ADD . /
WORKDIR /

RUN rm -rf tcdata/

CMD ["sh", "run.sh"]