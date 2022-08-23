hist = importdata("2dhist.txt");
fit = importdata('fitting.txt');
x= [0.01 : 0.01 : 0.6];
y = [130:-5:5];
imagesc(x,y, hist)
set(gca,'YDir','normal')
hold on
plot(fit(1,:), fit(2,:),'w','LineWidth',3)