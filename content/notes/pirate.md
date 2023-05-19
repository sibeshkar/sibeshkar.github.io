---
title : The Parable of The Robot Pirate
date : 10/05/2023
---
You're AI researcher Barbadosa the pirate trainer, and the first day at the job you have to teach a new robot pirate (a deep learning model, say, JackSparrowGPT), how to find treasure at sea. If you make one of these and make infinite copies of it, you will become very rich, very quick.  
  
Of course, you don't have a boat yet and have never actually ventured out to sea yourself, but you do have the entire island to yourselves. So much training data to train it using gradient descent. You also have infinite compute.
  
So this robot pirate, armed with a matter scanner, goes around the island inspecting every rock, every leaf, every tree.  
  
Sometimes it comes across a puddle and scans it. He puts his feet, records what wetness feels like, and steps out. "Oh this is a great example of a water body!", You exclaim. Surely this is meaningful progress at building treasure-finding systems. "I'm going to be very rich" you exclaim.  
  
It sometimes disturbs you how much resource JackSparrowGPT requires, unlike your own brain which runs on 20W of power and can learn from just a few samples. The robot pirate improves only linearly with exponential increase in data and compute you throw at it - but luckily, you have so much of it! Soon it has inspected and memorized single molecule and atom on the island. But these systems are huge and unwieldy to update, so you have to cutoff it's data at a certain date, it can't take in unlimited memory after all.  
  
You create new benchmarks to test these systems. For instance, the Tidal Wave benchmark, given the water level on Tuesday, it can predict the chances of high tide on Wednesday. And because the robot has years of data about tides, of course it can make a big guess, and he gets it right once, twice, thrice. You are impressed. The smartest guy you know can also make accurate tidal wave predictions. Maybe the robot is as smart as him.  
  
After years of training it on the dynamics of every single atom on the island, you discover it has completely beaten all the island-based benchmarks. It can beat the most talented island habitants at tree climbing, puddle hopping and stone throwing. You decide that it is now time.  
  
You point the robot towards the horizon and say : "You see that line in the distance. there are great treasures beyond that line. you have to go find them and bring them back"  
  
The robot promptly agrees. It starts walking towards the sea. For the first few steps, its just like walking on island, and the robot just beat the best human walker on the "Walking" benchmark last year. It continues to walk, until a wave comes in lifts it off its feet. There was never any water body or experience on the island similar to this, so the robot has no idea what to do. It errors out, spluttering and choking, as it drowns, and you watch on in horror.  

But surely you can augment it with information storage and retrieval, use it's vast understanding of things on the island to embed new data not seen before - and then search through it to make a new plan for the sea. Or - the robot understands "wetness" and "water body depth", because it has a embedded vector for "puddles" somewhere in it's latent space, based on it's understanding of puddles the island. You can use that to understand "the sea" as well.

You let it loose on the sea again - but alas, while the robot can vectorize 'water body depth' of the sea, there was a property "saline concentration" that it never encountered in the puddles on the island. As it goes deeper into the sea, the salt gets everywhere and the circuits malfunction. You watch in horror again as it drowns.

As it turns out, the sea (like real life) is an unbounded stream of unknown, unseen evaluation data, that a robot pirate bounded by scaled-up linear regression just cannot adapt to. It has to generalize in a provably unbounded / universal manner [[1]](/notes/unbounded)

You go back to the drawing board. Backpropagation is not the way. Gradient descent must be ditched. You `rm -rf` CUDA and start from scratch.

---
  
_This is the underwhelming state of modern machine learning, and surprisingly, what the hype / doomer AI cults are excited / worried about. There is little reason to get excited by / fear bounded robot pirates._