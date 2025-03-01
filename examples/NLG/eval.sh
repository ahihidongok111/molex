var="GPT2_M"

python src/gpt2_decode.py --vocab ./vocab \
--sample_file ./trained_models/GPT2_M/${var}/predict.${var}.jsonl \
--input_file ./data/e2e/test_formatted.jsonl --output_ref_file ${var}_ref.txt \
--output_pred_file ${var}_pred.txt

python eval/e2e/measure_scores.py ${var}_ref.txt ${var}_pred.txt -p