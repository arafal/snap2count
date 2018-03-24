from __future__ import unicode_literals

from django import forms

class ImageUploadForm(forms.Form):

	image = forms.ImageField()
