import pickle
import spacy
import re
import pdb




with open('data_200_session_1.pkl', 'rb') as file:
    article = pickle.load(file)

audio_rate=0.9

tts_flag = False
if tts_flag:
    from jxu.basiccmd.tts import *

else:
    from jxu.audio.audiosignal import *

if tts_flag:
    total_len = 0
    nlp = spacy.load("de_core_news_md")

    for i in range(1):
        tts(nlp, article[i], article_id=i, audio_rate=audio_rate)

else:
    for ind_censor_tag, censor_tag in enumerate(['VERB']):   # ['VERB', 'NOUN', 'ADJ', 'DET', 'ADV', 'AUX']
        for i in range(200):
            audio_to_chunk(article_id=i, audio_rate=audio_rate)
            beep_censoring(article_id=i, beep_word_type=censor_tag,  audio_rate=audio_rate)

import pickle
import spacy
import re
import pdb
from jxu.basiccmd.tts import *
nlp = spacy.load("de_core_news_md")
article= 'Das Experiment hat eine Woche lang gedauert. Viele Zuschauer sind gekommen und haben sich Theater-Stücke angeschaut. Das Experiment hat eine Woche lang gedauert.  '
tts_unshattered(nlp, article, article_id=0, file_root='/home/jxu/File/Data/NIBS/Stage_one/Audio/Database/Unshattered/',
                audio_rate=1.0, pitch=0.0, lang='de-DE', word_type='VERB', df_name='all_unshattered_ori_df.pkl')



pdb.set_trace()


import pandas as pd

all_df = pd.read_pickle('all_beep_df.pkl')

all_missing_df = all_df[[('SENTENCE_INFO', 'beeped_sen_content'),('SENTENCE_INFO', 'beep_word_type')]]
all_missing_df.to_pickle('all_missing_df.pkl')
all_missing_df.to_excel('all_question.xlsx')
#print(total_len)

"""
article = [''] * 111


article[0] = 'Das Experiment hat eine Woche lang gedauert. Viele Zuschauer sind gekommen und haben sich Theater-Stücke angeschaut.  '

article[1] = 'In dem Museum sollen wichtige Kunst-Werke hängen. In der National-Galerie ist zu wenig Platz für die wichtigen Kunst-Werke. Zwei Architekten haben das Museum entworfen.'

article[2] = 'Farben machen das Leben bunt. Der Regenbogen besteht aus vielen Farben. Obst und Gemüse ist oft sehr farbig. Und unsere Kleidung kann auch schön bunt sein. Viele Früchte sind rot, zum Beispiel Erdbeeren oder Himbeeren. '

article[3] = 'Viele Blumen sind rot, zum Beispiel Rosen. Gelb wie die Sonne sind auch Bananen, Zitronen oder Mais. Blau ist der Himmel bei schönem Wetter oder auch das Meer oder ein See.'

article[4] = 'Alle Farben zusammen ergeben schwarz. Das Gegenteil von schwarz ist weiß. Es gibt weiße Blumen, die Wolken und Schnee sind weiß. ' 



article[5] = 'Meine Familie ist groß, weil meine Eltern beide viele Geschwister haben.'

article[6] = 'Der Lehrer unterrichtet Schüler und bringt ihnen verschiedene Dinge bei. In einer Volksschule lehren sie die Kinder das Lesen und Schreiben. Lehrer arbeiten auch mit älteren Kindern und Jugendlichen. Dort unterrichten sie meistens ein bestimmtes Fach, Mathematik oder Sprachen zum Beispiel. Der Arzt behandelt kranke Leute in einer Praxis oder im Krankenhaus. Der Arzt verschreibt Medikamente oder andere Behandlungen. Es gibt viele verschiedene Ärzte, manche sind Chirurgen, andere sind Ohrenärzte oder Zahnärzte. Der Bäcker stellt Brot her. Bäcker können sehr viele verschiedene Sorten Brot machen, auch süßes Gebäck. Ein Koch arbeitet auch mit Lebensmitteln. In einem Restaurant bereitet ein Koch die Speisen zu. Ein Verkäufer arbeitet in einem Laden.' 

article[7] = 'Das Wetter in Deutschland ist vielseitig. Es gibt Wettervorhersagen im Fernsehen oder im Radio, die die Aussichten für die nächsten Tage liefern. Mit dem Beginn des Frühlings schmelzen das Eis und der Schnee, da die Temperaturen steigen und es warm wird. Im Frühling gibt es viele sonnige Tage. Die Sonne brennt im Sommer auf der Haut. Ab und zu blitzt und donnert es heftig, Gewitter ziehen auf.  Die Tage im Herbst sind oft windig. Wenn der Wind sehr stark bläst, entsteht ein Sturm. Mit dem Herbst bereitet sich die Natur wieder auf den Winter vor.'

article[8] = 'In Wien besuchen die Touristen am liebsten den Stephansdom und das Schloss Schönbrunn. Die Supermärkte in Österreich am Wochenende und in der Nacht nicht geöffnet sind.'

article[9] = 'Sie können durch neue Technologien leichter mit anderen kommunizieren, das heißt miteinander über Handys oder Internet Kontakt haben.'



article[10] = 'Dann dusche ich, zuerst ganz warm und am Schluss mit kaltem Wasser. '

article[11] = 'Einen Anzug tragen viele Leute bei der Arbeit. Zu einem Anzug gehören ein Gürtel und eine Krawatte. Eine Brille brauchen viele Leute, wenn sie schlecht sehen. Die Sonnenbrillen schützen die Augen vor der Sonne. An Regentagen benutzen die Leute einen Regenschirm.'

article[12] = 'In den Zeitungen, im Radio und im Fernsehen wird regelmäßig über Sport berichtet. Beim Fußball spielen zwei Mannschaften gegeneinander und versuchen, einen Ball in das Tor zu schießen. Ein Team besteht aus zehn Spielern und einem Tormann. Mehrere Schiedsrichter achten darauf, dass alle Spieler die Regeln einhalten. Beim Tennis spielen zwei gegeneinander. Beim Golf wird ein kleiner, harter Ball mit einem langen Schläger von einem Abschlagspunkt oft sehr weit über Hügel und Wiesen gespielt. Viele Deutsche gehen im Winter auch Ski fahren. Viele Deutschen fahren in die benachbarten Länder Österreich und Schweiz, dort gibt es sehr viele große Skigebiete.'

article[13] = 'Am Kopf finden wir die meisten Haare. Am Kopf sind mehrere Sinnesorgane: Augen, Ohren, Nase und Mund. Im Mund sind die Zähne und die Zunge. An den Händen haben wir zehn Finger, fünf Finger pro Hand. Am Fuß haben wir zehn Zehen. '

article[14] = 'Sie fällen erst viel Holz im Regen-Wald. Der Regen-Wald am Amazonas ist wichtig für die Erde. '





article[15] = 'Greta Thunberg will an einer Konferenz von der U N O teilnehmen. '

article[16] = 'Schuld sind vor allem die Menschen, denn die Menschen immer mehr Wald abholzen.'

article[17] = 'Das Internet ist ein welt-weites Netz aus Computern. Computer auf der ganzen Welt sind über viele Kabel miteinander verbunden. Die Handys und andere Geräte können sich mit dem Internet verbinden.'

article[18] = 'Astronaut fliegen mit Raketen in den Welt-Raum. '

article[19] = 'Die Hauptstadt von Spanien heißt Madrid. Die Hauptstadt von Österreich heißt Wien. Die Hauptstadt von Deutschland heißt Berlin. Die Hauptstadt von Frankreich heißt Paris. Die Hauptstadt von Italien heißt Rome. Die Menschen in Frankreich sprechen Französisch. Die Menschen in Spanien sprechen Spanien. Die Menschen in Österreich sprechen Deutsch. Die Menschen in Deutschland sprechen Deutsch. Die Menschen in Italien sprechen Italien.'


article[20] = 'Der Klima-Schutz soll verhindern, dass es auf der Erde immer wärmer wird. Ein wichtiges Mittel im Klima-Schutz ist, weniger Abgase zu produzieren. Die Abgase entstehen zum Beispiel beim Auto-Fahren, aber auch beim Heizen und in Kraft-Werken. Abgase schaden dem Klima. Das Klima-Schutz bedeutet zum Beispiel: Weniger Auto fahren, weniger Flugzeug fliegen, weniger Heizung und Strom verbrauchen.'

article[21] = 'Die Vereinten Nationen wollen den Frieden auf der Welt sichern. Die Vereinten Nationen wollen die Rechte der Menschen schützen.'

article[22] = 'Plastik-Müll belastet die Umwelt. Die Hersteller von Plastik-Teilen sollen etwas dafür bezahlen, wenn Strände saubergemacht werden müssen. Zur Umwelt gehören Wälder und Flüsse, Pflanzen und Tiere. Umwelt-Schutz bedeutet, die Natur möglichst wenig zu verschmutzen und sie nicht zu zerstören.'

article[23] = 'Wenn man Waren aus dem Ausland kauft, kann es sein, dass man dafür Zoll bezahlen muss.'

article[24] = 'Karneval wird jedes Jahr im Februar oder im März gefeiert.'





article[25] = 'Abschiebung bedeutet: einen Ausländer in sein Heimat-Land zurück-schicken.'

article[26] = 'Afrika liegt südlich von Europa.'

article[27] = 'Wenn der Akku aufgeladen ist, kann er Strom abgeben. Akkus gibt es zum Beispiel in Handys und Laptops. Mit einem Lade-Gerät kann man den Akku wieder aufladen.'

article[28] = 'In einem Foto-Album sammelt man zum Beispiel Fotos. Man sagt Album zu einer Musik-CD, damit ist gemeint, dass es eine Sammlung von Liedern ist.'

article[29] = 'Anonym ist ein anderes Wort für unbekannt oder „ohne Namen“. Die Autoren können anonym bleiben, wenn sie nicht bekannt werden wollen.' 




article[30] = 'Arbeitslos-sein bedeutet: man hat keinen Arbeits-Platz.'

article[31] = 'Architektinnen planen Gebäude.'

article[32] = 'Arm-sein bedeutet: wenig Geld zu haben.'

article[33] = 'Ein Autor ist ein Mensch, der Bücher oder andere Texte schreibt. Man nennt einen Autor auch Schriftsteller.'

article[34] = 'Bakterien sind sehr kleine Lebewesen. Manche Bakterien können Menschen krank machen.'





article[35] = 'Eine Bank ist eine Firma, die Geld leiht und verleiht. Menschen und Firmen können ein Konto bei der Bank eröffnen.'

article[36] = 'Biologie ist die Wissenschaft von den Lebewesen. Biologie gehört zu den Natur-Wissenschaften.'

article[37] = 'Ein Blogger oder eine Bloggerin sind Leute, die Texte schreiben und diese Texte im Internet veröffentlichen.'

article[38] = 'Eine Börse ist ein Markt für Aktien: Man kann dort Aktien kaufen oder verkaufen.'

article[39] = 'Brexit ist ein kurzes Wort für: Groß-Britannien tritt aus der Europäischen Union aus. Das Wort Brexit ist eigentlich aus 2 englischen Wörtern zusammen-gesetzt: British und exit.'


article[40] = 'In Deutschland und in Österreich heißen die Regierungs-Chefs Bundes-Kanzler. Die Kanzlerin oder der Kanzler wird vom Parlament gewählt. '

article[41] = 'Chemie ist die Wissenschaft von den verschiedenen Stoffen, aus denen die Dinge bestehen. Chemikerinnen und Chemiker erforschen, woraus etwas besteht. Chemikerinnen und Chemiker erforschen, woraus etwas besteht wie man neue Stoffe herstellen kann. '

article[42] = 'Kohlen-Dioxid entsteht zum Beispiel, wenn Holz oder Heizöl brennen. Das Treibhaus-Gas bedeutet, dass es zur Erd-Erwärmung beiträgt. Man nennt Klima-Wandel, dass die Temperaturen auf unserer Erde steigen.'

article[43] = 'Daten sind die Angaben über einen Menschen, die es im Internet gibt. Jeder Mensch hat ein Recht darauf, dass wichtige Daten über ihn geheim bleiben. '

article[44] = 'In einer Demokratie dürfen die Menschen frei wählen, wer das Land regieren soll. '





article[45] = 'Menschen mit einer Depression fühlen sich oft sehr traurig. '

article[46] = 'Der Euro ist das Geld, mit dem Menschen in Europa bezahlen. Die Abkürzung für Europäische Union ist EU.'

article[47] = 'Ein Experte oder eine Expertin kennt sich mit einer Sache besonders gut aus.'

article[48] = 'Ein Flüchtling ist jemand, der aus Not seine Heimat verlässt. '

article[49] = 'Gen-Technik ist eine Wissenschaft, die sich mit Genen beschäftigt. Mit Gen-Technik kann man die Gene von Menschen, Tieren oder Pflanzen verändern. '

article[50] = 'Ein Gift ist ein Stoff, der für Menschen gefährlich ist.'

article[51] = 'Google ist eine sehr große Internet-Firma.'

article[52] = 'Hamburg ist ein Stadtstaat und ein Land der Bundesrepublik Deutschland. Berlin ist ein Stadtstaat und ein Land der Bundesrepublik Deutschland. München ist ein Stadtstaat und ein Land der Bundesrepublik Deutschland. Wien ist ein Stadtstaat und ein Land der Bundesrepublik Österreich.'

article[53] = 'Handy ist ein anderes Wort für Mobil-Telefon.'

article[54] = 'Wenn ein Fluss Hoch-Wasser hat, fließt besonders viel Wasser darin. Hoch-Wasser gibt es zum Beispiel, wenn es längere Zeit sehr viel regnet.'



article[55] = 'Inflation bedeutet: Das Geld wird weniger wert. Inflation heißt: Wenn man etwas kaufen will, braucht man dafür immer mehr Geld. '

article[56] = 'Auf die eigene Seite von Instagram kann man Bilder und kurze Videos stellen. Man kann sich Bilder und Videos von anderen Menschen auf Instagram anschauen. '

article[57] = 'Juristen sind Menschen, die mit der Recht-Sprechung zu tun haben. '

article[58] = 'Klima ist ein Wort für das Wetter, das es an einem Ort viele Jahre lang gibt. '

article[59] = 'Auf der Konferenz treffen sich Menschen, um miteinander zu diskutieren.'





article[60] = 'Krebs ist eine schwere Krankheit.  '

article[61] = 'Ein Labor ist ein Raum, in dem Forscher wissenschaftliche Untersuchungen machen. '

article[62] = 'Lebensmittel sind die Dinge, die man essen kann.'

article[63] = 'Leichte Sprache ist eine Hilfe für Menschen, die keine schwierige Sprache verstehen. '

article[64] = 'Literatur bedeutet: Etwas, das man lesen kann. '



article[65] = 'Ein Manager ist ein leitender Angestellter in einer Firma. '

article[66] = 'Ein Marathon ist ein langer Lauf.'

article[67] = 'Eine Messe ist eine Ausstellung.'

article[68] = 'Ein Museum nennt man ein Gebäude, in dem Ausstellungen zu sehen sind.'

article[69] = 'Die Olympischen Spiele sind der größte Sport-Wettkampf von der Welt. '




article[70] = 'Physik versucht Regeln zu finden, nach denen die Dinge in der Welt funktionieren. Die Wissenschaftler aus der Physik heißen Physiker. '

article[71] = 'Unsere Erde ist ein Planet. Planeten kreisen um eine Sonne.'

article[72] = 'Presse ist ein anderes Wort für Zeitungen.'

article[73] = 'Psychiatrie ist die Wissenschaft von den geistigen und seelischen Krankheiten.'

article[74] = 'Ein anderes Wort für Raum-Fahrer heißt Astronaut. '



article[75] = 'Der Arzt schreibt auf den Rezept, welche Medizin ein Mensch braucht. Mit dem Rezept kann man die Medizin in der Apotheke kaufen.'

article[76] = 'Sauer-Stoff ist ein Gas. '

article[77] = 'Eine Statistik ist eine Sammlung von Zahlen zu einem bestimmten Thema. '

article[78] = 'Dieses Geld, das an den Staat abgegeben wird, nennt man Steuern. '

article[79] = 'Mit Strom kann man viele Geräte betreiben. '



article[80] = 'Ein Teleskop ist ein wissenschaftliches Instrument und man kann damit Dinge beobachten, die sehr weit entfernt sind. '

article[81] = 'Der Tierschutz zielt darauf ab, Tieren ein Leben ohne Leiden oder Schmerzen zu ermöglichen.'

article[82] = 'Während einer Trocken-Zeit regnet es sehr selten. '

article[83] = 'Uni ist das kurze Wort für Universität.  Eine Universität ist eine Hoch-Schule. '

article[84] = 'Veganer essen kein Fleisch und keine anderen Produkte von Tieren.'



article[85] = 'Ein Virus ist sehr klein, deshalb man es nicht mit einer Lupe sehenkann '

article[86] = 'Waren sind Dinge, die verkauft werden. '

article[87] = 'Washington ist die Hauptstadt von dem Land USA.'

article[88] = 'In einer Weltraum-Station können Raum-Fahrer leben und arbeiten. Eine Weltraum-Station kreist am Himmel um die Erde herum. '

article[89] = 'Auf der Youtube kann man Videos anschauen.'



article[90] = 'Das deutsche Alphabet beginnt mit A und endet mit Z. '

article[91] = 'Du bist krank und erhältst von deinem Arzt ein Rezept. Darauf hat der Arzt geschrieben, welche Medizin du brauchst. In der Apotheke gibst du der Apothekerin das Rezept. Sie weiß gut über Medikamente Bescheid. '

article[92] = 'An der Tankstelle füllen wir Benzin in den Tank. Mit der Benzin wird der Motor des Autos angetrieben.  Die Abgase der Benzinmotoren verschmutzen die Luft. '

article[93] = 'Brot kann man in der Bäckerei kaufen. '

article[94] = 'Auf einer CD sind der Musik oder gesprochener Text gespeichert. '



article[95] = 'Die Fische leben in Flüssen, Seen und Meeren. '

article[96] = 'Bei einigen isst man die Wurzel einer Pflanze, wie bei den Karotten. Beim Spinat und Kohl wird man die Blätter essen.'

article[97] = 'Ein Interview ist ein Gespräch mit Fragen und Antworten. '

article[98] = 'Wer Geld auf der Bank hat, besitzt dort ein Konto. Man kann seine Geld auf ein Sparkonto einzahlen, dafür man dann Zinsen bekommt. '

article[99] = 'Die Katzen jagen Mäuse.'



article[100] = 'Ein Kanal ist ein künstlicher Fluss. Kanäle sind auch Wasserstraßen für den Schiffs.'

article[101] = 'Einige Dinge strahlen selbst Licht ab.'

article[102] = 'Münzen sind das Kleingeld in der Geldbörse. Die kleinste Münze in Österreich ist ein Cent. Die größte Münze in Österreich ist zwei Euro. '

article[103] = 'Obst sollte man täglich essen, weil es wegen den Vitaminen sehr gesund ist.'

article[104] = 'Briefe und Pakete verschickt man mit der Post. '





article[105] = 'Einen Regenbogen sieht man vor, während oder nach einem Regen.'

article[106] = 'Der Treibstoff für Motoren lagert in der Tankstelle. '

article[107] = 'Eine Woche ist ein Zeitraum von sieben Tagen. Eine Woche beginnt mit dem Montag und endet am Sonntag. '

article[108] = 'Mit der Werbung machen die Firmen für die Waren aufmerksam.'

article[109] = 'In der Zeitung kann man die Neuigkeiten aus der Stadt und aus der Welt lesen. Journalisten schreiben Artikel in der Zeitung.  '


article[110] = 'Kuchen und Bonbons enthalten sehr viel Zucker. Zucker ist ein Nährstoff, der dem Körper Energie liefert.'

article[111] = 'Manche Menschen können die Texte in Zeitungen nicht verstehen.'

article[111] = ''

article[112] = ''

article[113] = ''

article[114] = ''





article[115] = ''

article[116] = ''

article[117] = ''

article[118] = ''

article[119] = ''



article[120] = ''

article[121] = ''

article[122] = ''

article[123] = ''

article[124] = ''





article[125] = ''

article[126] = ''

article[127] = ''

article[128] = ''

article[129] = ''



article[130] = ''

article[131] = ''

article[132] = ''

article[133] = ''

article[134] = ''





article[135] = ''

article[136] = ''

article[137] = ''

article[138] = ''

article[139] = ''

"""
