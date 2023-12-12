from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import text2emotion as te

sentences = [
    "A simple, but decent and well-managed apartment in an excellent location with easy access to everything in Warsaw. There is an almost direct bus to the Chopin airport. This is not the Hyatt or the Four Seasons, but it doesn't cost that much either.",
    "I did not like the way to pick up the keys. You have to go to a restaurant which is some 400 meters away from the apartment itself, just halway between two metro stations. So you will end up walking 10-15 minutes. This can be inconvenient if you are with heavier suitcases and/or the weather is extreme (heavy rain or heatwave).",
    "The location was great, I am quite a light sleeper and there were no loud noises that affected my sleep as the houses around protect from the busy streets. The view to the skyscrapers was great from the balcony, the apartment itself was comfortable, clean, nicely decorated, also looked pretty new, and everything was there that I needed. Would recommend!",
    "The unit did not match the description nor the photos in the ad. It was tiny, with no seating area as the description had claimed and it was supposed to be two bedrooms but it was one room with no door and a bed in the living room with no door. When we complained, they had the audacity to state that the description didn't mention doors! This still didn't explain the missing seating area. I would never use this company again!"
]

analyzer = SentimentIntensityAnalyzer()
for sentence in sentences:
    vs = analyzer.polarity_scores(sentence)
    print("\nOpinion: {}\n{}".format(sentence, str(vs)))
    print(te.get_emotion(sentence))
