import os

from paddle.trainer_config_helpers import *

def seq_to_seq_data(data_dir, is_generating, dict_size=30000,
                    subdir_structure=('train', 'dev', 'test'),
                    gen_result='test_output'):
    train_list, test_list, gen_list = map(lambda x: x+'.list', subdir_structure)

    src_lang_dict = os.path.join(data_dir, 'words.dict')
    trg_lang_dict = os.path.join(data_dir, 'tags.dict')
    with open(src_lang_dict, "r") as fin:
        src_dict = {word.strip():_id for _id, word in enumerate(fin)}
    with open(trg_lang_dict, "r") as fin:
        trg_dict = {word.strip():_id for _id, word in enumerate(fin)}

    if is_generating:
        train_list, trg_dict = None, None
        test_list = os.path.join(data_dir, gen_list)
    else:
        train_list = os.path.join(data_dir, train_list)
        test_list = os.path.join(data_dir,test_list)

    define_py_data_sources2(train_list, test_list,
                           module = "dataprovider", obj = "process",
                           args = {"src_dict": src_dict, "trg_dict": trg_dict})

    return {"src_dict_path": src_lang_dict, "trg_dict_path": trg_lang_dict,
            "gen_result": gen_result}

def gru_encoder_decoder(data_conf,
                        is_generating,
                        word_vector_dim=512,
                        encoder_size=512,
                        decoder_size=512,
                        beam_size=3,
                        max_length=250):
    for k, v in data_conf.iteritems():
        globals()[k] = v
    source_dict_dim = len(open(src_dict_path, "r").readlines())
    target_dict_dim = len(open(trg_dict_path, "r").readlines())
    gen_trans_file = gen_result

    if decoder_size % 3 != 0:
        decoder_size = decoder_size / 3 * 3

    src_word_id = data_layer(name='source_language_word', size=source_dict_dim)
    src_embedding = embedding_layer(
        input=src_word_id,
        size=word_vector_dim,
        param_attr=ParamAttr(name='_source_language_embedding'))
    src_forward = simple_gru(input=src_embedding, size=encoder_size)
    src_backward = simple_gru(input=src_embedding,
                              size=encoder_size,
                              reverse=True)
    encoded_vector = concat_layer(input=[src_forward, src_backward])

    with mixed_layer(size=decoder_size) as encoded_proj:
        encoded_proj += full_matrix_projection(input=encoded_vector)

    wtf = data_layer(name='wtf', size=source_dict_dim)
    src_forward_wtf = simple_gru(input=src_embedding, size=encoder_size)
    src_backward_wtf = simple_gru(input=src_embedding,
                              size=encoder_size,
                              reverse=True)
    encoded_wtf_vec = concat_layer(input=[src_forward, src_backward])

    with mixed_layer(size=decoder_size) as encoded_wtf:
        encoded_wtf_proj += full_matrix_projection(input=wtf)

    backward_first = first_seq(input=[src_backward, src_backward_wtf])
    with mixed_layer(size=decoder_size,
                     act=TanhActivation(), ) as decoder_boot:
        decoder_boot += full_matrix_projection(input=backward_first)

    def gru_decoder_with_attention(enc_vec, enc_proj, enc_wtf_vec, enc_wtf_proj, current_word):
        decoder_mem = memory(name='gru_decoder',
                             size=decoder_size,
                             boot_layer=decoder_boot)

        context = simple_attention(encoded_sequence=enc_vec,
                                   encoded_proj=enc_proj,
                                   decoder_state=decoder_mem, )


        context = simple_attention(encoded_sequence=enc_wtf_vec,
                                   encoded_proj=enc_wtf_proj,
                                   decoder_state=decoder_mem, )


        with mixed_layer(size=decoder_size * 3) as decoder_inputs:
            decoder_inputs += full_matrix_projection(input=context)
            decoder_inputs += full_matrix_projection(input=context2)
            decoder_inputs += full_matrix_projection(input=current_word)


        gru_step = gru_step_layer(name='gru_decoder',
                                  input=decoder_inputs,
                                  output_mem=decoder_mem,
                                  size=decoder_size)

        with mixed_layer(size=target_dict_dim,
                         bias_attr=True,
                         act=SoftmaxActivation()) as out:
            out += full_matrix_projection(input=gru_step)
        return out

    decoder_group_name = "decoder_group"
    group_inputs=[StaticInput(input=encoded_vector, is_seq=True),
                  StaticInput(input=encoded_proj, is_seq=True)]

    if not is_generating:
        trg_embedding = embedding_layer(
            input=data_layer(name='target_language_word',
                             size=target_dict_dim),
            size=word_vector_dim,
            param_attr=ParamAttr(name='_target_language_embedding'))
        group_inputs.append(trg_embedding)

        decoder = recurrent_group(name=decoder_group_name,
                                  step=gru_decoder_with_attention,
                                  input=group_inputs)

        lbl = data_layer(name='target_language_next_word',
                         size=target_dict_dim)
        cost = classification_cost(input=decoder, label=lbl)

        outputs(cost)
    else:
        # In generation, the decoder predicts a next target word based on
        # the encoded source sequence and the last generated target word.

        # The encoded source sequence (encoder's output) must be specified by
        # StaticInput, which is a read-only memory.
        # Embedding of the last generated word is automatically gotten by
        # GeneratedInputs, which is initialized by a start mark, such as <s>,
        # and must be included in generation.

        trg_embedding = GeneratedInput(
            size=target_dict_dim,
            embedding_name='_target_language_embedding',
            embedding_size=word_vector_dim)
        group_inputs.append(trg_embedding)

        beam_gen = beam_search(name=decoder_group_name,
                               step=gru_decoder_with_attention,
                               input=group_inputs,
                               bos_id=0,
                               eos_id=1,
                               beam_size=beam_size,
                               max_length=max_length)

        seqtext_printer_evaluator(input=beam_gen,
                                  id_input=data_layer(name="sent_id", size=1),
                                  dict_file=trg_dict_path,
                                  result_file=gen_trans_file)
        outputs(beam_gen)
