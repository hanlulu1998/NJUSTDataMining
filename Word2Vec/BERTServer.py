# 启动BERT服务端
from bert_serving.server import BertServer
from bert_serving.server.helper import get_args_parser


def startBERT():
    # 设置服务端参数
    # -model_dir 预训练文件的路径
    # -port 输入端口号码
    # -port_out 输出端口号
    # -max_seq_len 句子最大长度
    # -cpu 使用tensorflow的CPU版本进行处理
    args = get_args_parser().parse_args(['-model_dir', r'.\uncased_L-12_H-768_A-12',
                                         '-port', '86500',
                                         '-port_out', '86501',
                                         '-max_seq_len', '512',
                                         '-mask_cls_sep',
                                         '-cpu'])
    # 开启服务
    bs = BertServer(args)
    bs.start()


if __name__ == "__main__":
    startBERT()
