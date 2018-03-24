from __future__ import unicode_literals

from django.db import models

# Create your models here.

class ImageModel(models.Model):

	model = models.ImageField(upload_to = 'images/', default = 'media/None/no-img.jpg')
	created = models.DateTimeField(auto_now_add = True)