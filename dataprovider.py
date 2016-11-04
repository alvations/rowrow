from paddle.trainer.PyDataProvider2 import *

UNK_IDX = 2
START = "<s>"
END = "<e>"


def hook(settings, words_dict, tags_dict, file_list, **kwargs):
    # job_mode = 1: training mode
    # job_mode = 0: generating mode
    settings.job_mode = tags_dict is not None
    settings.src_dict = words_dict
    settings.logger.info("Vocabulary dict len : %d" % (len(settings.words_dict)))
    settings.sample_count = 0

    if settings.job_mode:
        settings.trg_dict = tags_dict
        settings.slots = [
            integer_value_sequence(len(settings.words_dict)),
            integer_value_sequence(len(settings.tags_dict)),
            integer_value_sequence(len(settings.tags_dict)),
        ]
        settings.logger.info("Tagset dict len : %d" % (len(settings.tags_dict)))

def _get_ids(s, dictionary):
    return [dictionary[START]] + \
           [dictionary.get(w, UNK_IDX) for w in words.strip().split()] + \
           [dictionary[END]]


@provider(init_hook=hook, pool_size=500)
def process(settings, file_name):
    with open(file_name, 'r') as f:
        for line_count, line in enumerate(f):
            line_split = line.strip().split('\t')
            if settings.job_mode and len(line_split) != 2:
                continue
            word_seq = line_split[0]  # one sentence sequence
            word_ids = _get_ids(word_seq, settings.words_dict)

            if settings.job_mode:
                tag_seq = line_split[1]  # one POS tag sequence
                tag_ids = [settings.trg_dict.get(w, UNK_IDX)
                           for tag in tag_seq.strip().split()]

                # remove sequence whose length > 80 in training mode
                if len(word_ids) > 80 or len(tag_ids) > 80:
                    continue

                tag_ids_next = tag_ids + [settings.trg_dict[END]]
                tag_ids = [settings.trg_dict[START]] + tag_ids
                yield word_ids, tag_ids, tag_ids_next

            else:
                yield word_ids, [line_count]
