from django import forms

class TextForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea(attrs={'class':'form-control','rows':10,'cols':100}))