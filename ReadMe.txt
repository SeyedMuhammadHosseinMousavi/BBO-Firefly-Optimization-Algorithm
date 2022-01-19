%% Hybrid Biogeography Based Optimization Firefly Algorithm - Created in 19 Jan 2022 by Seyed Muhammad Hossein Mousavi
% The following lines of code, extracts 4 signal features, namely
% 'EnergyEntropy', 'ShortTimeEnergy', 'SpectralCentroid' and 'SpectralFlux'
% which 'ShortTimeEnergy' is considered to be target and others as inputs. 
% Fuzzy logic creates initial model to fit. Evolutionary training has two stages:
% first training using BBO algorithm in order to created first fuzzy
% evolutionary model and second stage is fitting BBO model to better
% condition by Firefly algorithm. Obviously you can use your signal, audio
% or even matrix and customize your parameters based on your system and data.
% ------------------------------------------------ 
% Feel free to contact me if you find any problem using the code: 
% Author: SeyedMuhammadHosseinMousavi
% My Email: mosavi.a.i.buali@gmail.com 
% My Google Scholar: https://scholar.google.com/citations?user=PtvQvAQAAAAJ&hl=en 
% My GitHub: https://github.com/SeyedMuhammadHosseinMousavi?tab=repositories 
% My ORCID: https://orcid.org/0000-0001-6906-2152 
% My Scopus: https://www.scopus.com/authid/detail.uri?authorId=57193122985 
% My MathWorks: https://www.mathworks.com/matlabcentral/profile/authors/9763916#
% my RG: https://www.researchgate.net/profile/Seyed-Mousavi-17
% ------------------------------------------------ 
% Hope it help you, enjoy the code and wish me luck :)