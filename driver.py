import extractor
import visualizer

#sims = extractor5.Extractor("C:/Users/bradn/Downloads/Resume&Job_Description/data/")

sims = extractor5_trigrams.Extractor("file path here")

#sims = extractor5.Extractor("C:/Users/bradn/Documents/Presentation_Docs/")

sims.analyze(n=10, topics=25)
