import pandas as pd
from df_utils import *  # kaggle util functions


def append_data(df, file_path):
    pass


if __name__ == "__main__":
    # ! The current columns in the final csv: text, generated
    # mistral 7b
    print("Model Mistral")
    mistral_7b_cme_v7_df = pd.read_csv("Mistral7B_CME_v7.csv")
    mistral_7b_cme_v7_df["model_source"] = "mistral"
    mistral_df_to_append = mistral_7b_cme_v7_df[["text", "generated"]]
    get_df_info(mistral_7b_cme_v7_df)

    # gemini pro
    print("Model Gemini")
    gemini_pro_df = pd.read_csv("gemini_train_essays_v1.csv")
    gemini_pro_df.rename(
        columns={"source": "model_source", "label": "generated"}, inplace=True
    )
    gemini_pro_df_to_append = gemini_pro_df[["text", "generated"]]
    get_df_info(gemini_pro_df)

    # llama 7b
    print("Model Llama")
    llama_df = pd.read_csv("llama_essays_a.csv")
    llama_df_2 = pd.read_csv("llama_essays_b.csv")
    llama_df_3 = pd.read_csv("llama_essays_c.csv")
    llama_df = pd.concat([llama_df, llama_df_2, llama_df_3], ignore_index=True)
    llama_df["model_source"] = "llama"
    llama_df_to_append = llama_df[["text", "generated"]]
    get_df_info(llama_df)

    # human written corpus along with the generated corpus
    print("Mixed Corpus")
    mixed_df = pd.read_csv("train_essays_RDizzl3_seven_v2.csv")
    mixed_df.rename(columns={"label": "generated"}, inplace=True)
    mixed_df_to_append = mixed_df[["text", "generated"]]
    get_df_info(mixed_df)
    # single out the human written corpus
    print("Human Written Corpus")
    human_df = mixed_df[mixed_df["generated"] == 0]
    get_df_info(human_df)

    # TODO: human written corpus load - unrelated to the topic

    # TODO: machine written corpus load - unrelated to the topic

    # TODO: the text processing part

    # combine the two dataframes - the resulting columns [text, generated] - training part

    # the original training data
    train_essays_df = pd.read_csv("../train_essays.csv")
    train_essays_df = train_essays_df[["text", "generated"]]

    # final df to be returned
    print("Final Corpus to be returned")
    final_df = pd.concat(
        [
            gemini_pro_df_to_append,
            llama_df_to_append,
            mixed_df_to_append,
            train_essays_df,
        ],
        ignore_index=True,
    )
    get_df_info(final_df)
    final_df.to_csv("final_data_df.csv", index=False)
