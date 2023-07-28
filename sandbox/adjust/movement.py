run = sample.runs[0]

s_img = run.get_spectral_image(wavelength=500)
m_img = run.get_label_image("g")

fig, ax = plt.subplots(1, 2)
ax[0].imshow(s_img, cmap="gray", vmin=0, vmax=255)
ax[1].imshow(m_img, cmap="gray", vmin=0, vmax=255)
plt.show()

s_imgs = run.get_spectral_images()

fig, ax = plt.subplots()


ims = []
for i, img in enumerate(s_imgs):
    im = ax.imshow(img, cmap="gray", animated=True)
    if i == 0:
        ax.imshow(img, cmap="gray")  # show an initial one first
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=75, blit=True)
ani.save("movie.mp4")


plt.show()
