Compile:

./compile.sh


Usage:

Prepare an estimated score (e.g. 008_EST.xml) and a groud-truth score (e.g. 008_GT.xml). Then use the following command:

./evaluate_XML_voicePlus.sh 008_GT 008_EST ER

The evaluation metrics will be output in ER.txt.
The output metrics are, from left to right

(1)  pitch error rate (%)
(2)  missing note rate (%)
(3)  extra note rate (%)
(4)  onset time error rate (%)
(5)  offset time error rate (%)
(6)  mean of (1) to (5)
(7)  voice error rate (%)
(8)  mean of (1) to (5) and (7)
(9)  voice precision (%)
(10) voice recall (%)
(11) voice F measure (%)
(12) note value scale error
(13) hand error rate (%)

Details of error analysis are output in 008_EST_err_detail.txt, where all detected errors and note-wise alignment information are given. Note IDs are described by symbols of the form PXX-YY-ZZ, which indicates the ZZ-th note in the YY-th bar in part XX of the MusicXML file. "GtID" refers to the ground-truth MusicXML file and "EstID" refers to the estimated MusicXML file.


Remark:

When there are many errors in the estimated MusicXML file, there can be a significant amount of errors in the alignment process, which will influence the error analysis. You can view the alignment result in the following way. After running the evaluation script as

./evaluate_XML_voicePlus.sh 008_GT 008_EST ER

you will find files 008_GT_fmt3x.txt and 008_EST_auto_match.txt. Please visit

https://midialignment.github.io/score-performance-match-editor/ScorePerformanceMatchEditor.html

and open the two files in the browser. You can view the alignment result and check how pitch errors, extra notes, and missing notes were identified.


Update history:

(2022/Jan/27) Some internal modules were updated.
(2022/Jan/18) Added an output file with details of error analysis. Some internal modules were modified.
(2021/Dec/17) Fixed somes bugs.

Old versions can be accessed from the following link:

https://github.com/amtevaluation/amtevaluation.github.io


References:

Please see the following papers for the definitionos of the metrics.

Eita Nakamura, Emmanouil Benetos, Kazuyoshi Yoshii, Simon Dixon,
Towards Complete Polyphonic Music Transcription: Integrating Multi-Pitch Detection and Rhythm Quantization. 
Proc. 43rd IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 101-105, 2018.

Yuki Hiramatsu, Eita Nakamura, Kazuyoshi Yoshii,
Joint Estimation of Note Values and Voices for Audio-to-Score Piano Transcription
Proc. 22nd International Society for Music Information Retrieval Conference (ISMIR), 2021.

Contact:

For any inquiries, please contact
Eita Nakamura (eita.nakamura@gmail.com)
