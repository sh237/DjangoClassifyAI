# Generated by Django 4.1.3 on 2022-11-26 07:17

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Photo",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("image", models.ImageField(upload_to="photos")),
                ("uploaded_at", models.DateTimeField(auto_now_add=True, null=True)),
                ("result_sentences", models.TextField(blank=True, null=True)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="user_photos",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
    ]
