from pathlib import Path

DATASET_NAME = 'synthetic0'
FOLD_COUNT = 10
CLASS_COUNT = 5


def main():
    ughz = """<?xml version="1.0" encoding="utf-8"?>""" + '\n'
    ughz += """<labels xmlns="http://mulan.sourceforge.net/labels">""" + '\n'
    for class_nr in range(CLASS_COUNT):
        ughz += f"""<label name="Class{class_nr}"></label>""" + '\n'
    ughz += """</labels>"""

    for fold_nr in range(FOLD_COUNT):
        xml_path = Path.cwd() / 'folds' / str(fold_nr) / 'mulan.xml'
        xml_path.write_text(ughz)


if __name__ == '__main__':
    main()
